"""Export experiment summaries to JSON, CSV, and Rich tables."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

from rich.table import Table

from llm_phase_bench.console import console

if TYPE_CHECKING:
    from pathlib import Path

    from llm_phase_bench.reporting.aggregator import ExperimentSummary


def save_summary_json(summaries: list[ExperimentSummary], path: Path) -> None:
    """Save experiment summaries as JSON."""
    data = [s.model_dump() for s in summaries]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    console.print(f"[green]JSON saved:[/] {path}")


def save_summary_csv(summaries: list[ExperimentSummary], path: Path) -> None:
    """Save experiment summaries as flat CSV for spreadsheet analysis."""
    if not summaries:
        return

    fieldnames = [
        "experiment_name",
        "model_id",
        "quantization",
        "attn_implementation",
        "length_group",
        "sample_count",
        "em",
        "f1",
        "ttft_mean_ms",
        "ttft_median_ms",
        "ttft_p95_ms",
        "tpot_mean_ms",
        "tpot_median_ms",
        "tpot_p95_ms",
        "decode_tok_per_sec_mean",
        "peak_memory_mb",
        "prefill_peak_memory_mb",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "experiment_name": s.experiment_name,
                    "model_id": s.model_id,
                    "quantization": s.quantization,
                    "attn_implementation": s.attn_implementation,
                    "length_group": s.length_group,
                    "sample_count": s.sample_count,
                    "em": f"{s.quality.mean_em:.4f}",
                    "f1": f"{s.quality.mean_f1:.4f}",
                    "ttft_mean_ms": f"{s.ttft.mean:.2f}",
                    "ttft_median_ms": f"{s.ttft.median:.2f}",
                    "ttft_p95_ms": f"{s.ttft.p95:.2f}",
                    "tpot_mean_ms": f"{s.tpot.mean:.2f}" if s.tpot else "",
                    "tpot_median_ms": f"{s.tpot.median:.2f}" if s.tpot else "",
                    "tpot_p95_ms": f"{s.tpot.p95:.2f}" if s.tpot else "",
                    "decode_tok_per_sec_mean": (
                        f"{s.decode_tokens_per_sec.mean:.2f}"
                        if s.decode_tokens_per_sec
                        else ""
                    ),
                    "peak_memory_mb": (
                        f"{s.peak_memory_mb:.1f}" if s.peak_memory_mb else ""
                    ),
                    "prefill_peak_memory_mb": (
                        f"{s.prefill_peak_memory_mb:.1f}"
                        if s.prefill_peak_memory_mb
                        else ""
                    ),
                }
            )

    console.print(f"[green]CSV saved:[/] {path}")


def print_summary_table(summaries: list[ExperimentSummary]) -> None:
    """Print a Rich table comparing all experiments."""
    table = Table(title="Benchmark Results", show_lines=True)

    table.add_column("Model", style="cyan")
    table.add_column("Quant")
    table.add_column("Attn")
    table.add_column("Length")
    table.add_column("N", justify="right")
    table.add_column("EM", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("TTFT\n(ms)", justify="right")
    table.add_column("TPOT\n(ms)", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("Mem\n(MB)", justify="right")

    for s in summaries:
        table.add_row(
            s.model_id.split("/")[-1],
            s.quantization,
            s.attn_implementation,
            s.length_group,
            str(s.sample_count),
            f"{s.quality.mean_em:.3f}",
            f"{s.quality.mean_f1:.3f}",
            f"{s.ttft.median:.1f}",
            f"{s.tpot.median:.1f}" if s.tpot else "-",
            f"{s.decode_tokens_per_sec.mean:.1f}" if s.decode_tokens_per_sec else "-",
            f"{s.peak_memory_mb:.0f}" if s.peak_memory_mb else "-",
        )

    console.print(table)
