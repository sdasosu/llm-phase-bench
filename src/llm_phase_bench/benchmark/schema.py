"""Shared data schemas for benchmark results (no heavy dependencies)."""

from __future__ import annotations

from pydantic import BaseModel


class SampleResult(BaseModel, frozen=True):
    """Result of running inference on a single sample."""

    sample_id: str
    raw_prompt: str
    reference_answers: list[str]
    prediction: str
    num_generated_tokens: int
    actual_prompt_tokens: int
    ttft_ms: float
    tpot_ms: float | None
    decode_tokens_per_sec: float | None
    peak_memory_mb: float | None
    prefill_peak_memory_mb: float | None
