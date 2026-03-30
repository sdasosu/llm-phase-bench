"""Compute median and IQR for TTFT and TPOT from raw JSONL benchmark data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_phase_bench.utils.jsonl import load_jsonl_raw

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class TimingIQR:
    """Median and interquartile range for a timing metric."""

    median: float
    q1: float
    q3: float
    iqr: float


@dataclass(frozen=True)
class ConfigStats:
    """Per-config statistics for TTFT and TPOT."""

    platform: str
    config: str
    n: int
    ttft: TimingIQR
    tpot: TimingIQR


def load_timing_values(jsonl_path: Path) -> dict[str, list[float]]:
    """Load TTFT and TPOT values from a JSONL file.

    Args:
        jsonl_path: Path to a JSONL file with benchmark results.

    Returns:
        Dict with "ttft" and "tpot" lists of float values.
    """
    ttft_vals: list[float] = []
    tpot_vals: list[float] = []
    for row in load_jsonl_raw(jsonl_path):
        ttft = row["ttft_ms"]
        tpot = row["tpot_ms"]
        if isinstance(ttft, (int, float)):
            ttft_vals.append(ttft)
        if isinstance(tpot, (int, float)):
            tpot_vals.append(tpot)
    return {"ttft": ttft_vals, "tpot": tpot_vals}


def compute_iqr(values: list[float]) -> TimingIQR:
    """Compute median and IQR from a list of values.

    Args:
        values: Non-empty list of float measurements.

    Returns:
        TimingIQR with median, Q1, Q3, and IQR.
    """
    import numpy as np

    arr = np.array(values)
    q1, median, q3 = np.percentile(arr, [25, 50, 75]).tolist()
    return TimingIQR(median=median, q1=q1, q3=q3, iqr=q3 - q1)


def collect_stats(
    directory: Path,
    *,
    platform: str,
    prefix: str = "",
) -> list[ConfigStats]:
    """Collect IQR statistics for all JSONL files in a directory.

    Args:
        directory: Directory containing JSONL benchmark output files.
        platform: Platform label for the results.
        prefix: Optional filename prefix filter (e.g. "cuda-sdpa-fp16").

    Returns:
        List of ConfigStats, one per JSONL file.
    """
    results: list[ConfigStats] = []
    for jsonl_path in sorted(directory.glob("*.jsonl")):
        if prefix and not jsonl_path.stem.startswith(prefix):
            continue
        metrics = load_timing_values(jsonl_path)
        results.append(
            ConfigStats(
                platform=platform,
                config=jsonl_path.stem,
                n=len(metrics["ttft"]),
                ttft=compute_iqr(metrics["ttft"]),
                tpot=compute_iqr(metrics["tpot"]),
            )
        )
    return results


def print_stats_table(stats: list[ConfigStats]) -> None:
    """Print a formatted table of timing statistics.

    Args:
        stats: List of ConfigStats to display.
    """
    from rich.table import Table

    from llm_phase_bench.console import console

    table = Table(title="Timing Statistics (Median & IQR)", show_lines=True)
    table.add_column("Platform", style="cyan")
    table.add_column("Config")
    table.add_column("N", justify="right")
    table.add_column("TTFT med (ms)", justify="right")
    table.add_column("TTFT IQR (ms)", justify="right")
    table.add_column("TPOT med (ms)", justify="right")
    table.add_column("TPOT IQR (ms)", justify="right")

    for r in stats:
        table.add_row(
            r.platform,
            r.config,
            str(r.n),
            f"{r.ttft.median:.1f}",
            f"{r.ttft.iqr:.2f}",
            f"{r.tpot.median:.1f}",
            f"{r.tpot.iqr:.2f}",
        )
    console.print(table)

    console.print("\n[bold]Summary for paper[/]")
    platforms = sorted({r.platform for r in stats})
    for platform in platforms:
        plat_stats = [r for r in stats if r.platform == platform]
        ttft_iqrs = [r.ttft.iqr for r in plat_stats]
        tpot_iqrs = [r.tpot.iqr for r in plat_stats]
        console.print(f"\n[bold cyan]{platform}:[/]")
        console.print(
            f"  TTFT IQR range: {min(ttft_iqrs):.2f} - {max(ttft_iqrs):.2f} ms"
        )
        console.print(
            f"  TPOT IQR range: {min(tpot_iqrs):.2f} - {max(tpot_iqrs):.2f} ms"
        )
