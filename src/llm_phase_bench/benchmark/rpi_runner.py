"""Raspberry Pi benchmark using llama-cpp-python.

Runs the same SQuAD QA benchmark as the GPU pipeline, but using
llama.cpp GGUF models on CPU.  Reads a pre-built manifest.json and
outputs JSONL in the same SampleResult schema as the GPU runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_phase_bench.benchmark.schema import SampleResult

if TYPE_CHECKING:
    from llama_cpp import Llama  # ty: ignore[unresolved-import]


@dataclass(frozen=True)
class GgufModelSpec:
    """GGUF model file mapping."""

    friendly_name: str
    gguf_f16: str
    gguf_q4km: str


MODELS: dict[str, GgufModelSpec] = {
    "qwen3.5-0.8b": GgufModelSpec(
        friendly_name="qwen3.5-0.8b",
        gguf_f16="qwen3.5-0.8b-f16.gguf",
        gguf_q4km="qwen3.5-0.8b-q4km.gguf",
    ),
    "gemma-3-1b": GgufModelSpec(
        friendly_name="gemma-3-1b",
        gguf_f16="gemma-3-1b-it-f16.gguf",
        gguf_q4km="gemma-3-1b-it-q4km.gguf",
    ),
    "llama-3.2-1b": GgufModelSpec(
        friendly_name="llama-3.2-1b",
        gguf_f16="llama-3.2-1b-f16.gguf",
        gguf_q4km="llama-3.2-1b-q4km.gguf",
    ),
}

LENGTH_GROUPS: dict[str, int] = {
    "short": 8,
    "medium": 32,
    "long": 64,
}

WARMUP_RUNS = 3
WARMUP_PROMPT = "What is 1+1? Answer briefly."


def _current_rss_mb() -> float:
    """Current RSS in MiB via /proc/self/status (Linux only)."""
    status = Path("/proc/self/status").read_text()
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return 0.0


def run_single(
    llm: Llama,
    sample: dict[str, Any],
    max_tokens: int,
) -> SampleResult:
    """Run inference on a single sample, extracting llama.cpp timings.

    Args:
        llm: A loaded Llama model instance.
        sample: Dict with "raw_prompt", "sample_id", "reference_answers".
        max_tokens: Maximum tokens to generate.

    Returns:
        SampleResult with timing and quality data.
    """
    from llama_cpp import llama_cpp  # ty: ignore[unresolved-import]

    raw_prompt = sample["raw_prompt"]

    llama_cpp.llama_perf_context_reset(llm._ctx.ctx)  # noqa: SLF001

    messages = [{"role": "user", "content": raw_prompt}]
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    prediction = response["choices"][0]["message"]["content"] or ""

    perf = llama_cpp.llama_perf_context(llm._ctx.ctx)  # noqa: SLF001

    prompt_eval_ms = perf.t_p_eval_ms
    n_prompt = perf.n_p_eval
    eval_ms = perf.t_eval_ms
    n_eval = perf.n_eval

    ttft_ms = prompt_eval_ms
    tpot_ms: float | None = None
    decode_tps: float | None = None

    if n_eval > 1:
        tpot_ms = eval_ms / (n_eval - 1)
        decode_tps = (n_eval - 1) / (eval_ms / 1000.0)

    return SampleResult(
        sample_id=sample["sample_id"],
        raw_prompt=raw_prompt,
        reference_answers=sample["reference_answers"],
        prediction=prediction,
        num_generated_tokens=n_eval,
        actual_prompt_tokens=n_prompt,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        decode_tokens_per_sec=decode_tps,
        peak_memory_mb=_current_rss_mb(),
        prefill_peak_memory_mb=None,
    )


def warmup(llm: Llama) -> None:
    """Run warmup inferences to stabilize timing.

    Args:
        llm: A loaded Llama model instance.
    """
    for _ in range(WARMUP_RUNS):
        llm.create_chat_completion(
            messages=[{"role": "user", "content": WARMUP_PROMPT}],
            max_tokens=4,
            temperature=0.0,
        )
    print(f"  Warmup: {WARMUP_RUNS} runs completed")


def run_experiment(
    model_path: Path,
    friendly_name: str,
    quant_label: str,
    manifest: dict[str, Any],
    output_dir: Path,
    n_ctx: int = 512,
    limit: int | None = None,
) -> None:
    """Run a full experiment for one model+quantization combo.

    Args:
        model_path: Path to the GGUF model file.
        friendly_name: Short name for the model (used in output filenames).
        quant_label: Quantization label (e.g. "f16", "q4km").
        manifest: Loaded manifest dict with "length_groups" key.
        output_dir: Directory to write JSONL results to.
        n_ctx: Context window size for llama.cpp.
        limit: Max samples per length group (None = all).
    """
    from llama_cpp import Llama  # ty: ignore[unresolved-import]

    print(f"\n{'=' * 60}")
    print(f"Model: {friendly_name} ({quant_label})")
    print(f"GGUF:  {model_path}")
    print(f"{'=' * 60}")

    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=4,
        verbose=False,
    )
    warmup(llm)

    for group_name, max_tokens in LENGTH_GROUPS.items():
        samples = manifest["length_groups"].get(group_name, [])
        if limit is not None:
            samples = samples[:limit]
        if not samples:
            print(f"  Skipping {group_name}: no samples")
            continue

        out_file = output_dir / f"{friendly_name}_{quant_label}_{group_name}.jsonl"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{group_name}] {len(samples)} samples, max_tokens={max_tokens}")

        with out_file.open("w") as f:
            for i, sample in enumerate(samples):
                result = run_single(llm, sample, max_tokens)
                f.write(result.model_dump_json() + "\n")
                f.flush()

                if (i + 1) % 10 == 0 or i == len(samples) - 1:
                    tpot_str = f"TPOT={result.tpot_ms:.1f}ms" if result.tpot_ms else ""
                    print(
                        f"    [{i + 1}/{len(samples)}] "
                        f"TTFT={result.ttft_ms:.1f}ms {tpot_str}"
                    )

        print(f"  Saved: {out_file}")

    del llm


def run_rpi_benchmark(
    manifest_path: Path,
    models_dir: Path,
    output_dir: Path,
    *,
    model_keys: list[str] | None = None,
    quants: list[str] | None = None,
    limit: int | None = None,
    n_ctx: int = 512,
) -> None:
    """Run RPi LLM benchmark.

    Args:
        manifest_path: Path to manifest.json from data preparation.
        models_dir: Directory containing GGUF files.
        output_dir: Output directory for JSONL results.
        model_keys: Which models to benchmark (default: all).
        quants: Which quantization levels to test (default: ["f16", "q4km"]).
        limit: Limit samples per length group (None = all).
        n_ctx: Context window size.
    """
    if model_keys is None:
        model_keys = list(MODELS.keys())
    if quants is None:
        quants = ["f16", "q4km"]

    manifest = json.loads(manifest_path.read_text())
    print(f"Manifest: {manifest['total_samples']} samples")

    for model_key in model_keys:
        spec = MODELS[model_key]
        for quant in quants:
            gguf_name = spec.gguf_f16 if quant == "f16" else spec.gguf_q4km
            model_path = models_dir / gguf_name
            if not model_path.exists():
                print(f"SKIP: {model_path} not found")
                continue
            run_experiment(
                model_path=model_path,
                friendly_name=spec.friendly_name,
                quant_label=quant,
                manifest=manifest,
                output_dir=output_dir,
                n_ctx=n_ctx,
                limit=limit,
            )
