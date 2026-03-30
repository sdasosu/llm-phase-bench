"""JSONL read/write utilities (SSOT for line-delimited JSON I/O)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from llm_phase_bench.benchmark.schema import SampleResult

if TYPE_CHECKING:
    from pathlib import Path


def load_jsonl_results(path: Path) -> list[SampleResult]:
    """Load sample results from a JSONL file.

    Args:
        path: Path to a JSONL file containing serialized SampleResult objects.

    Returns:
        List of validated SampleResult instances.
    """
    results: list[SampleResult] = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                results.append(SampleResult.model_validate(json.loads(stripped)))
    return results


def load_jsonl_raw(path: Path) -> list[dict[str, object]]:
    """Load raw dicts from a JSONL file without schema validation.

    Args:
        path: Path to a JSONL file.

    Returns:
        List of raw dicts, one per non-empty line.
    """
    rows: list[dict[str, object]] = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows
