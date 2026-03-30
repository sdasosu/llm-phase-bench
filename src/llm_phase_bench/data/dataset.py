"""SQuAD dataset loading, token-length filtering, and manifest creation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from transformers import PreTrainedTokenizerBase

from llm_phase_bench.console import console

if TYPE_CHECKING:
    from llm_phase_bench.config.schema import BenchmarkConfig, LengthGroup


class ManifestSample(TypedDict):
    """A single sample in the dataset manifest."""

    sample_id: str
    context: str
    question: str
    raw_prompt: str
    reference_answers: list[str]
    length_group: str
    token_counts: dict[str, int]


class DatasetManifest(TypedDict):
    """The full dataset manifest written to disk."""

    length_groups: dict[str, list[ManifestSample]]
    model_ids: list[str]
    total_samples: int


@dataclass(frozen=True)
class _RawSample:
    """Internal representation of a SQuAD sample before filtering."""

    sample_id: str
    context: str
    question: str
    raw_prompt: str
    reference_answers: list[str]


@dataclass(frozen=True)
class _FilteredCandidate:
    """A sample that passed token-length filtering for all tokenizers."""

    raw: _RawSample
    token_counts: dict[str, int] = field(default_factory=dict)


def _load_tokenizers(model_ids: list[str]) -> dict[str, PreTrainedTokenizerBase]:
    """Load tokenizers for all models."""
    from transformers import AutoTokenizer

    tokenizers: dict[str, PreTrainedTokenizerBase] = {}
    for model_id in model_ids:
        console.print(f"  Loading tokenizer: [cyan]{model_id}[/]")
        tok = AutoTokenizer.from_pretrained(model_id)
        assert isinstance(tok, PreTrainedTokenizerBase)
        tokenizers[model_id] = tok
    return tokenizers


def _count_tokens(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> int:
    """Count tokens for a text using a specific tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _filter_candidates(
    raw_samples: list[_RawSample],
    tokenizers: dict[str, PreTrainedTokenizerBase],
    length_group: LengthGroup,
) -> list[_FilteredCandidate]:
    """Filter samples that fall within tolerance for ALL tokenizers."""
    target = length_group.prompt_tokens
    tolerance = length_group.tolerance
    lo, hi = target - tolerance, target + tolerance

    candidates: list[_FilteredCandidate] = []
    per_tokenizer_counts = dict.fromkeys(tokenizers, 0)

    for sample in raw_samples:
        token_counts: dict[str, int] = {}
        all_in_range = True

        for model_id, tokenizer in tokenizers.items():
            count = _count_tokens(tokenizer, sample.raw_prompt)
            token_counts[model_id] = count
            if lo <= count <= hi:
                per_tokenizer_counts[model_id] += 1
            else:
                all_in_range = False

        if all_in_range:
            candidates.append(_FilteredCandidate(raw=sample, token_counts=token_counts))

    console.print(
        f"  Length group [bold]{length_group.name}[/] (target={target} ±{tolerance}):"
    )
    for model_id, count in per_tokenizer_counts.items():
        console.print(f"    {model_id}: {count} candidates in range")
    console.print(f"    Intersection: {len(candidates)} candidates")

    return candidates


def _collect_all_model_ids(config: BenchmarkConfig) -> list[str]:
    """Extract unique model IDs from all experiment groups."""
    seen: set[str] = set()
    model_ids: list[str] = []
    for group in config.experiment_groups:
        for model in group.models:
            if model.model_id not in seen:
                seen.add(model.model_id)
                model_ids.append(model.model_id)
    return model_ids


def prepare_dataset(config: BenchmarkConfig) -> Path:
    """Prepare a shared dataset manifest filtered by token length.

    Loads SQuAD v1.1 validation, tokenizes each candidate with ALL model
    tokenizers, keeps only samples within tolerance for all, and saves a
    reusable JSON manifest.

    Args:
        config: The benchmark configuration.

    Returns:
        Path to the saved manifest file.
    """
    import datasets

    from llm_phase_bench.data.prompting import format_qa_prompt

    console.print("[bold]Preparing dataset manifest...[/]")

    model_ids = _collect_all_model_ids(config)
    console.print(f"Models: {model_ids}")

    # Load SQuAD v1.1 validation
    console.print("Loading SQuAD v1.1 validation split...")
    squad = datasets.load_dataset("rajpurkar/squad", split="validation")

    # Build raw samples
    raw_samples: list[_RawSample] = [
        _RawSample(
            sample_id=row["id"],
            context=row["context"],
            question=row["question"],
            raw_prompt=format_qa_prompt(
                context=row["context"],
                question=row["question"],
            ),
            reference_answers=row["answers"]["text"],
        )
        for row in squad
    ]
    console.print(f"Total SQuAD samples: {len(raw_samples)}")

    # Load tokenizers
    console.print("Loading tokenizers...")
    tokenizers = _load_tokenizers(model_ids)

    # Filter and sample per length group
    rng = random.Random(config.generation.seed)  # noqa: S311
    manifest_groups: dict[str, list[ManifestSample]] = {}

    for lg in config.length_groups:
        candidates = _filter_candidates(raw_samples, tokenizers, lg)

        if len(candidates) < lg.num_samples:
            msg = (
                f"Length group '{lg.name}' needs {lg.num_samples} samples "
                f"but only {len(candidates)} candidates passed the "
                f"3-tokenizer intersection filter "
                f"(target={lg.prompt_tokens} ±{lg.tolerance} tokens)."
            )
            raise RuntimeError(msg)

        selected = rng.sample(candidates, lg.num_samples)

        manifest_groups[lg.name] = [
            ManifestSample(
                sample_id=c.raw.sample_id,
                context=c.raw.context,
                question=c.raw.question,
                raw_prompt=c.raw.raw_prompt,
                reference_answers=c.raw.reference_answers,
                length_group=lg.name,
                token_counts=c.token_counts,
            )
            for c in selected
        ]
        console.print(f"  [green]✓[/] {lg.name}: selected {lg.num_samples} samples")

    # Save manifest
    cache_dir = Path(config.paths.data_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    total = sum(len(samples) for samples in manifest_groups.values())
    manifest: DatasetManifest = {
        "length_groups": manifest_groups,
        "model_ids": model_ids,
        "total_samples": total,
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    console.print(f"[bold green]Manifest saved:[/] {manifest_path} ({total} samples)")

    return manifest_path


def load_manifest(path: Path) -> DatasetManifest:
    """Load a previously saved dataset manifest."""
    return json.loads(path.read_text())
