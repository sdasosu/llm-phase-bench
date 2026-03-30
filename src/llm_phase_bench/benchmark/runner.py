"""Single-sample inference runner with timing instrumentation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
import transformers
from rich.progress import Progress

from llm_phase_bench.benchmark.schema import SampleResult
from llm_phase_bench.benchmark.timing import FirstTokenTimer
from llm_phase_bench.console import console
from llm_phase_bench.data.prompting import apply_chat_template

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from llm_phase_bench.data.dataset import ManifestSample

MS_PER_SEC = 1000.0
BYTES_PER_MB = 1024 * 1024
WARMUP_RUNS = 3
WARMUP_PROMPT = "Hello"
WARMUP_MAX_TOKENS = 1


def _is_cuda(device: torch.device) -> bool:
    return device.type == "cuda"


def run_single_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sample: ManifestSample,
    max_new_tokens: int,
    seed: int,
) -> SampleResult:
    """Run inference on a single sample with timing instrumentation.

    Args:
        model: The loaded model.
        tokenizer: The model's tokenizer.
        sample: A manifest sample with prompt and reference answers.
        max_new_tokens: Maximum tokens to generate.
        seed: Random seed for reproducibility.

    Returns:
        A SampleResult with timing and quality data.
    """
    device = model.device
    use_cuda = _is_cuda(device)

    # Apply chat template and tokenize
    chat_prompt = apply_chat_template(tokenizer, sample["raw_prompt"])
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    actual_prompt_tokens = inputs["input_ids"].shape[1]

    # Set seed for reproducibility
    transformers.set_seed(seed)

    # Reset memory tracking
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    # Create timer hook and start events
    timer = FirstTokenTimer()
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None

    if use_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    start_time = time.perf_counter()

    # Generate with LogitsProcessor hook
    with torch.inference_mode():
        outputs = model.generate(  # ty: ignore[call-non-callable]
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=[timer],
        )

    # Record end time (synchronize only AFTER generation completes)
    if use_cuda:
        assert end_event is not None
        end_event.record()
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    # Decode output (only the generated tokens)
    generated_ids = outputs[0, actual_prompt_tokens:]
    prediction: str = tokenizer.decode(generated_ids, skip_special_tokens=True)  # ty: ignore[invalid-assignment]
    num_generated = len(generated_ids)

    # Compute timing metrics
    if timer.uses_cuda_events:
        assert start_event is not None
        assert timer.first_token_event is not None
        assert end_event is not None
        ttft_ms = start_event.elapsed_time(timer.first_token_event)
        total_ms = start_event.elapsed_time(end_event)
        decode_ms = total_ms - ttft_ms
    else:
        assert timer.first_token_time is not None
        ttft_ms = (timer.first_token_time - start_time) * MS_PER_SEC
        decode_ms = (end_time - timer.first_token_time) * MS_PER_SEC

    tpot_ms: float | None = None
    decode_tps: float | None = None
    if num_generated > 1:
        tpot_ms = decode_ms / (num_generated - 1)
        decode_tps = (num_generated - 1) / (decode_ms / MS_PER_SEC)

    # Memory metrics
    peak_mem: float | None = None
    if use_cuda:
        peak_mem = torch.cuda.max_memory_allocated() / BYTES_PER_MB

    return SampleResult(
        sample_id=sample["sample_id"],
        raw_prompt=sample["raw_prompt"],
        reference_answers=sample["reference_answers"],
        prediction=prediction,
        num_generated_tokens=num_generated,
        actual_prompt_tokens=actual_prompt_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        decode_tokens_per_sec=decode_tps,
        peak_memory_mb=peak_mem,
        prefill_peak_memory_mb=None,
    )


def _warmup(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Run warmup inferences to stabilize CUDA kernel timing."""
    device = model.device
    inputs = tokenizer(WARMUP_PROMPT, return_tensors="pt").to(device)
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model.generate(**inputs, max_new_tokens=WARMUP_MAX_TOKENS)  # ty: ignore[call-non-callable]
    if _is_cuda(device):
        torch.cuda.synchronize()
    console.print(f"  Warmup: {WARMUP_RUNS} runs completed")


def run_experiment(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: list[ManifestSample],
    max_new_tokens: int,
    seed: int,
    output_path: Path,
) -> list[SampleResult]:
    """Run inference on all samples and write results incrementally to JSONL.

    Args:
        model: The loaded model.
        tokenizer: The model's tokenizer.
        samples: List of manifest samples.
        max_new_tokens: Maximum tokens to generate per sample.
        seed: Random seed.
        output_path: Path to the JSONL output file.

    Returns:
        List of all sample results.
    """
    _warmup(model, tokenizer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[SampleResult] = []

    with (
        output_path.open("a") as f,
        Progress(console=console) as progress,
    ):
        task = progress.add_task("Inference", total=len(samples))
        for sample in samples:
            result = run_single_inference(
                model,
                tokenizer,
                sample,
                max_new_tokens,
                seed,
            )
            # Append-only JSONL with per-sample flush
            f.write(result.model_dump_json() + "\n")
            f.flush()
            results.append(result)
            progress.advance(task)

    return results
