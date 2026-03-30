"""Full benchmark sweep orchestration."""

from __future__ import annotations

import datetime
import json
from typing import TYPE_CHECKING

from llm_phase_bench.benchmark.runner import run_experiment
from llm_phase_bench.console import console
from llm_phase_bench.data.dataset import load_manifest
from llm_phase_bench.models.loader import load_model, unload_model

if TYPE_CHECKING:
    from pathlib import Path

    from llm_phase_bench.config.schema import BenchmarkConfig


def run_benchmark(
    config: BenchmarkConfig,
    output_dir: Path,
    *,
    experiment_filter: str | None = None,
    model_filter: str | None = None,
    limit: int | None = None,
) -> Path:
    """Execute the full benchmark sweep.

    Loads each model once per experiment group, runs all length groups,
    then unloads before loading the next model.

    Args:
        config: The benchmark configuration.
        output_dir: Base directory for run output.
        experiment_filter: Only run experiments matching this name.
        model_filter: Only run models matching this friendly name.
        limit: Limit samples per length group (for quick testing).

    Returns:
        Path to the run directory containing all JSONL results.
    """
    # Create timestamped run directory
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    config_snapshot = run_dir / "config.json"
    config_snapshot.write_text(
        json.dumps(config.model_dump(mode="json"), indent=2, ensure_ascii=False)
    )

    # Load manifest
    manifest_path = config.paths.data_cache_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    console.print(f"[bold]Loaded manifest:[/] {manifest['total_samples']} samples")

    for group in config.experiment_groups:
        if experiment_filter and experiment_filter not in group.name:
            console.print(f"[dim]Skipping experiment: {group.name}[/]")
            continue

        console.print(f"\n[bold blue]═══ Experiment: {group.name} ═══[/]")

        for model_spec in group.models:
            if model_filter and model_filter not in model_spec.friendly_name:
                console.print(f"[dim]Skipping model: {model_spec.friendly_name}[/]")
                continue

            console.print(f"\n[bold cyan]── Model: {model_spec.friendly_name} ──[/]")

            # Load model with experiment-level config
            model, tokenizer = load_model(
                model_id=model_spec.model_id,
                device=group.device,
                quantization=group.quantization,
                attn_implementation=group.attn_implementation,
            )

            for lg in config.length_groups:
                samples = manifest["length_groups"].get(lg.name, [])
                if limit is not None:
                    samples = samples[:limit]

                if not samples:
                    console.print(
                        f"  [yellow]No samples for length group: {lg.name}[/]"
                    )
                    continue

                console.print(
                    f"\n  [bold]Length group: {lg.name}[/] "
                    f"({len(samples)} samples, max_new_tokens={lg.output_tokens})"
                )

                # JSONL output path
                jsonl_name = f"{group.name}_{model_spec.friendly_name}_{lg.name}.jsonl"
                jsonl_path = run_dir / jsonl_name

                run_experiment(
                    model=model,
                    tokenizer=tokenizer,
                    samples=samples,
                    max_new_tokens=lg.output_tokens,
                    seed=config.generation.seed,
                    output_path=jsonl_path,
                )

            unload_model(model)

    console.print(f"\n[bold green]Benchmark complete.[/] Results: {run_dir}")
    return run_dir
