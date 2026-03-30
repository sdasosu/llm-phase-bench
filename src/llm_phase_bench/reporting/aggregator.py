"""Aggregate raw JSONL results into summary statistics."""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from pydantic import BaseModel

from llm_phase_bench.benchmark.metrics import compute_metrics

if TYPE_CHECKING:
    from llm_phase_bench.benchmark.schema import SampleResult


class TimingStats(BaseModel, frozen=True):
    """Statistical summary for a timing metric."""

    mean: float
    median: float
    std: float
    p95: float


class QualityMetrics(BaseModel, frozen=True):
    """Quality metric summary."""

    mean_em: float
    mean_f1: float


class ExperimentSummary(BaseModel, frozen=True):
    """Aggregated results for one experiment configuration + length group."""

    experiment_name: str
    model_id: str
    quantization: str
    attn_implementation: str
    length_group: str
    sample_count: int
    quality: QualityMetrics
    ttft: TimingStats
    tpot: TimingStats | None
    decode_tokens_per_sec: TimingStats | None
    peak_memory_mb: float | None
    prefill_peak_memory_mb: float | None


def _compute_timing_stats(values: list[float]) -> TimingStats:
    """Compute timing statistics from a list of values."""
    sorted_vals = sorted(values)
    p95_idx = int(len(sorted_vals) * 0.95)
    return TimingStats(
        mean=statistics.mean(values),
        median=statistics.median(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        p95=sorted_vals[min(p95_idx, len(sorted_vals) - 1)],
    )


def aggregate_experiment(
    results: list[SampleResult],
    experiment_name: str,
    model_id: str,
    quantization: str,
    attn_implementation: str,
    length_group: str,
) -> ExperimentSummary:
    """Aggregate a list of sample results into an experiment summary.

    Args:
        results: Per-sample inference results.
        experiment_name: Name of the experiment group.
        model_id: Model identifier.
        quantization: Quantization mode used.
        attn_implementation: Attention backend used.
        length_group: Length group name.

    Returns:
        Aggregated experiment summary with quality and timing stats.
    """
    # Quality metrics
    predictions = [r.prediction for r in results]
    references = [r.reference_answers for r in results]
    metrics = compute_metrics(predictions, references)

    # Timing stats
    ttft_values = [r.ttft_ms for r in results]
    ttft_stats = _compute_timing_stats(ttft_values)

    tpot_stats: TimingStats | None = None
    tpot_values = [r.tpot_ms for r in results if r.tpot_ms is not None]
    if tpot_values:
        tpot_stats = _compute_timing_stats(tpot_values)

    decode_tps_stats: TimingStats | None = None
    decode_tps_values = [
        r.decode_tokens_per_sec for r in results if r.decode_tokens_per_sec is not None
    ]
    if decode_tps_values:
        decode_tps_stats = _compute_timing_stats(decode_tps_values)

    # Memory (max across samples)
    peak_mems = [r.peak_memory_mb for r in results if r.peak_memory_mb is not None]
    prefill_mems = [
        r.prefill_peak_memory_mb
        for r in results
        if r.prefill_peak_memory_mb is not None
    ]

    return ExperimentSummary(
        experiment_name=experiment_name,
        model_id=model_id,
        quantization=quantization,
        attn_implementation=attn_implementation,
        length_group=length_group,
        sample_count=len(results),
        quality=QualityMetrics(
            mean_em=metrics["mean_em"],
            mean_f1=metrics["mean_f1"],
        ),
        ttft=ttft_stats,
        tpot=tpot_stats,
        decode_tokens_per_sec=decode_tps_stats,
        peak_memory_mb=max(peak_mems) if peak_mems else None,
        prefill_peak_memory_mb=max(prefill_mems) if prefill_mems else None,
    )
