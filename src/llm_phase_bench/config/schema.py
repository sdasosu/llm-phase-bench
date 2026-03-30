"""Pydantic models for benchmark experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class ModelSpec(BaseModel, frozen=True):
    """A model to benchmark."""

    model_id: str
    friendly_name: str


class LengthGroup(BaseModel, frozen=True):
    """Prompt/output token length bucket."""

    name: str
    prompt_tokens: int
    output_tokens: int
    num_samples: int
    tolerance: int


class GenerationParams(BaseModel, frozen=True):
    """Fixed generation parameters for reproducibility."""

    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    seed: int = 42


class ExperimentGroup(BaseModel, frozen=True):
    """One experiment configuration (device + quantization + attention)."""

    name: str
    device: Literal["cuda", "mps", "cpu"]
    models: list[ModelSpec]
    quantization: Literal["fp16", "int4"]
    attn_implementation: Literal["sdpa", "flash_attention_2"]


class PathsConfig(BaseModel, frozen=True):
    """Output and cache directory paths."""

    output_dir: Path = Path("artifacts/runs")
    data_cache_dir: Path = Path("artifacts/data")


class BenchmarkConfig(BaseModel, frozen=True):
    """Top-level benchmark configuration."""

    experiment_groups: list[ExperimentGroup]
    length_groups: list[LengthGroup]
    generation: GenerationParams = GenerationParams()
    paths: PathsConfig = PathsConfig()
