"""CLI entry point for the LLM inference benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from llm_phase_bench.console import console

if TYPE_CHECKING:
    from llm_phase_bench.config.schema import BenchmarkConfig


def _load_config(path: Path) -> BenchmarkConfig:
    """Load and validate a BenchmarkConfig from a YAML file."""
    import yaml

    from llm_phase_bench.config import BenchmarkConfig as _BenchmarkConfig

    raw = yaml.safe_load(path.read_text())
    return _BenchmarkConfig(**raw)


def cmd_run(args: argparse.Namespace) -> None:
    """Execute a benchmark sweep."""
    from llm_phase_bench.benchmark.orchestrator import run_benchmark

    config = _load_config(args.config)
    output_dir = args.output_dir or Path(config.paths.output_dir)

    run_benchmark(
        config=config,
        output_dir=output_dir,
        experiment_filter=args.experiments,
        model_filter=args.models,
        limit=args.limit,
    )


def cmd_prepare_data(args: argparse.Namespace) -> None:
    """Prepare the shared dataset manifest."""
    from llm_phase_bench.data.dataset import prepare_dataset

    config = _load_config(args.config)
    prepare_dataset(config)


def cmd_report(args: argparse.Namespace) -> None:
    """Generate summary report from raw JSONL results."""
    from llm_phase_bench.reporting.aggregator import aggregate_experiment
    from llm_phase_bench.reporting.export import (
        print_summary_table,
        save_summary_csv,
        save_summary_json,
    )
    from llm_phase_bench.utils.jsonl import load_jsonl_results

    run_dir = Path(args.run_dir)
    jsonl_files = sorted(run_dir.glob("*.jsonl"))

    if not jsonl_files:
        console.print(f"[red]No JSONL files found in {run_dir}[/]")
        sys.exit(1)

    summaries = []
    for jsonl_path in jsonl_files:
        results = load_jsonl_results(jsonl_path)
        if not results:
            continue

        # Parse experiment metadata from filename:
        # {experiment_name}_{model_name}_{length_group}.jsonl
        stem = jsonl_path.stem
        parts = stem.rsplit("_", maxsplit=1)
        length_group = parts[-1] if len(parts) > 1 else "unknown"
        experiment_model = parts[0] if len(parts) > 1 else stem

        # Match filename against config to extract full metadata
        experiment_name = experiment_model
        model_id = ""
        quantization = ""
        attn_impl = ""

        config_path = run_dir / "config.json"
        if config_path.exists():
            import json

            config_data = json.loads(config_path.read_text())
            for group in config_data.get("experiment_groups", []):
                for model in group.get("models", []):
                    candidate = f"{group['name']}_{model['friendly_name']}"
                    if experiment_model == candidate:
                        experiment_name = group["name"]
                        model_id = model["model_id"]
                        quantization = group.get("quantization", "")
                        attn_impl = group.get("attn_implementation", "")
                        break

        summary = aggregate_experiment(
            results=results,
            experiment_name=experiment_name,
            model_id=model_id,
            quantization=quantization,
            attn_implementation=attn_impl,
            length_group=length_group,
        )
        summaries.append(summary)

    if not summaries:
        console.print("[red]No results to aggregate[/]")
        sys.exit(1)

    # Output
    print_summary_table(summaries)
    save_summary_json(summaries, run_dir / "summary.json")
    save_summary_csv(summaries, run_dir / "summary.csv")


def cmd_dry_run(args: argparse.Namespace) -> None:
    """Validate config without running inference."""
    from llm_phase_bench.utils.device import device_info

    config = _load_config(args.config)
    console.print("[bold green]Config validated successfully.[/]")
    console.print(config)

    console.print("\n[bold]Device info:[/]")
    for key, value in device_info().items():
        console.print(f"  {key}: {value}")

    # Summarize experiment matrix
    total_experiments = sum(
        len(g.models) * len(config.length_groups) for g in config.experiment_groups
    )
    console.print(f"\n[bold]Experiment matrix:[/] {total_experiments} total runs")
    for group in config.experiment_groups:
        console.print(
            f"  {group.name}: {len(group.models)} models x "
            f"{len(config.length_groups)} length groups"
        )


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate paper figures from benchmark results."""
    from llm_phase_bench.reporting.plotting import generate_figures

    generate_figures(
        gpu_dir=args.gpu_dir,
        cpu_dir=args.cpu_dir,
        rpi_dir=args.rpi_dir,
        output_dir=args.output_dir,
    )
    console.print(f"[green]Figures saved to:[/] {args.output_dir}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Compute and display timing statistics (median/IQR)."""
    from llm_phase_bench.reporting.statistics import collect_stats, print_stats_table

    all_stats = []
    if args.gpu_dir:
        for prefix in ("cuda-sdpa-fp16", "cuda-sdpa-int4", "cuda-fa2-fp16"):
            all_stats.extend(collect_stats(args.gpu_dir, platform="GPU", prefix=prefix))
    if args.cpu_dir:
        all_stats.extend(
            collect_stats(args.cpu_dir, platform="CPU", prefix="cpu-sdpa-fp16")
        )
    if args.rpi_dir:
        all_stats.extend(collect_stats(args.rpi_dir, platform="RPi"))

    if not all_stats:
        console.print("[red]No data directories specified.[/]")
        sys.exit(1)

    print_stats_table(all_stats)


def cmd_rpi_run(args: argparse.Namespace) -> None:
    """Run RPi benchmark using llama.cpp GGUF models."""
    from llm_phase_bench.benchmark.rpi_runner import run_rpi_benchmark

    run_rpi_benchmark(
        manifest_path=args.manifest,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        model_keys=args.models or None,
        quants=args.quants or None,
        limit=args.limit,
        n_ctx=args.n_ctx,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="LLM inference benchmark framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- run --
    run_parser = subparsers.add_parser("run", help="Execute a benchmark sweep")
    run_parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )
    run_parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override output directory"
    )
    run_parser.add_argument(
        "--experiments", type=str, default=None, help="Filter by experiment name"
    )
    run_parser.add_argument(
        "--models", type=str, default=None, help="Filter by model name"
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples per length group",
    )
    run_parser.set_defaults(func=cmd_run)

    # -- prepare-data --
    prep_parser = subparsers.add_parser(
        "prepare-data", help="Prepare shared dataset manifest"
    )
    prep_parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )
    prep_parser.set_defaults(func=cmd_prepare_data)

    # -- report --
    report_parser = subparsers.add_parser(
        "report", help="Generate summary from raw results"
    )
    report_parser.add_argument(
        "--run-dir", type=Path, required=True, help="Path to run directory"
    )
    report_parser.set_defaults(func=cmd_report)

    # -- dry-run --
    dry_parser = subparsers.add_parser("dry-run", help="Validate config only")
    dry_parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML config"
    )
    dry_parser.set_defaults(func=cmd_dry_run)

    # -- plot --
    plot_parser = subparsers.add_parser("plot", help="Generate paper figures")
    plot_parser.add_argument(
        "--gpu-dir", type=Path, required=True, help="GPU benchmark results directory"
    )
    plot_parser.add_argument(
        "--cpu-dir", type=Path, required=True, help="CPU benchmark results directory"
    )
    plot_parser.add_argument(
        "--rpi-dir", type=Path, required=True, help="RPi benchmark results directory"
    )
    plot_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures",
    )
    plot_parser.set_defaults(func=cmd_plot)

    # -- stats --
    stats_parser = subparsers.add_parser(
        "stats", help="Compute timing statistics (median/IQR)"
    )
    stats_parser.add_argument(
        "--gpu-dir", type=Path, default=None, help="GPU benchmark results directory"
    )
    stats_parser.add_argument(
        "--cpu-dir", type=Path, default=None, help="CPU benchmark results directory"
    )
    stats_parser.add_argument(
        "--rpi-dir", type=Path, default=None, help="RPi benchmark results directory"
    )
    stats_parser.set_defaults(func=cmd_stats)

    # -- rpi-run --
    rpi_parser = subparsers.add_parser("rpi-run", help="Run RPi benchmark (llama.cpp)")
    rpi_parser.add_argument(
        "--manifest", type=Path, required=True, help="Path to manifest.json"
    )
    rpi_parser.add_argument(
        "--models-dir", type=Path, required=True, help="Directory with GGUF files"
    )
    rpi_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/rpi"),
        help="Output directory for JSONL results",
    )
    rpi_parser.add_argument(
        "--models",
        nargs="*",
        choices=["qwen3.5-0.8b", "gemma-3-1b", "llama-3.2-1b"],
        default=None,
        help="Which models to benchmark",
    )
    rpi_parser.add_argument(
        "--quants",
        nargs="*",
        choices=["f16", "q4km"],
        default=None,
        help="Which quantization levels to test",
    )
    rpi_parser.add_argument(
        "--limit", type=int, default=None, help="Limit samples per length group"
    )
    rpi_parser.add_argument(
        "--n-ctx", type=int, default=512, help="Context window size"
    )
    rpi_parser.set_defaults(func=cmd_rpi_run)

    return parser


def main() -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)
