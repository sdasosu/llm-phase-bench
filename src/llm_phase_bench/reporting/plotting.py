"""Generate paper figures from benchmark results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_phase_bench.benchmark.metrics import exact_match
from llm_phase_bench.utils.jsonl import load_jsonl_raw

if TYPE_CHECKING:
    from pathlib import Path

    import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
    import pandas as pd

MODEL_LABELS = {
    "qwen3.5-0.8b": "Qwen3.5\n0.8B",
    "gemma-3-1b": "Gemma 3\n1B",
    "llama-3.2-1b": "Llama 3.2\n1B",
}

LENGTH_ORDER = ["short", "medium", "long"]
LENGTH_TOKENS = {"short": 64, "medium": 128, "long": 256}


def load_results(results_dir: Path, platform: str) -> list[dict]:
    """Load all JSONL files into a flat list of records.

    Args:
        results_dir: Directory containing JSONL benchmark output files.
        platform: Platform label ("gpu", "cpu", or "rpi").

    Returns:
        List of dicts with platform, model, config, length, and metrics.
    """
    rows: list[dict] = []
    for jsonl_path in sorted(results_dir.glob("*.jsonl")):
        parts = jsonl_path.stem.split("_")
        if platform in ("gpu", "cpu"):
            group, model, length = parts[0], parts[1], parts[2]
            group_parts = group.split("-", 1)[1]
            attn, quant = group_parts.rsplit("-", 1)
            config = f"{attn}+{quant}".upper()
        else:
            model, quant, length = parts[0], parts[1], parts[2]
            config = quant.upper()

        for r in load_jsonl_raw(jsonl_path):
            prediction = str(r.get("prediction", ""))
            ref_answers = r.get("reference_answers", [])
            assert isinstance(ref_answers, list)
            em = max(
                (exact_match(prediction, str(ref)) for ref in ref_answers),
                default=0.0,
            )
            rows.append(
                {
                    "platform": platform,
                    "model": model,
                    "config": config,
                    "length": length,
                    "ttft_ms": r["ttft_ms"],
                    "tpot_ms": r["tpot_ms"],
                    "em": em,
                    "mem_mb": r.get("peak_memory_mb"),
                }
            )
    return rows


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute median TTFT/TPOT per group."""
    return (
        df.groupby(["platform", "model", "config", "length"], observed=True)
        .agg(
            ttft_median=("ttft_ms", "median"),
            tpot_median=("tpot_ms", "median"),
            em_mean=("em", "mean"),
            mem_median=("mem_mb", "median"),
        )
        .reset_index()
    )


def plot_tpot_delta(summary: pd.DataFrame) -> plt.Figure:
    """Figure: TPOT % change from quantization on GPU vs RPi, per model.

    Args:
        summary: DataFrame from ``build_summary``.

    Returns:
        Matplotlib Figure with 3 subplots (one per model).
    """
    import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
    import pandas as pd
    import seaborn as sns  # ty: ignore[unresolved-import]

    records: list[dict] = []
    for model in MODEL_LABELS:
        for length in LENGTH_ORDER:
            for plat, base_cfg, accel_cfg, label in (
                ("gpu", "SDPA+FP16", "SDPA+INT4", "GPU: INT4"),
                ("rpi", "F16", "Q4KM", "RPi: Q4_K_M"),
            ):
                mask = (
                    (summary["platform"] == plat)
                    & (summary["model"] == model)
                    & (summary["length"] == length)
                )
                baseline = summary[mask & (summary["config"] == base_cfg)][
                    "tpot_median"
                ]
                accel = summary[mask & (summary["config"] == accel_cfg)]["tpot_median"]
                if len(baseline) and len(accel):
                    b = baseline.to_numpy()[0]
                    a = accel.to_numpy()[0]
                    records.append(
                        {
                            "model": model,
                            "length": length,
                            "platform": label,
                            "delta_pct": (a - b) / b * 100,
                        }
                    )

    delta_df = pd.DataFrame(records)
    models = list(MODEL_LABELS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for idx, model in enumerate(models):
        ax = axes[idx]
        data = delta_df[delta_df["model"] == model]
        sns.barplot(
            data=data,
            x="length",
            y="delta_pct",
            hue="platform",
            palette=["#e74c3c", "#2ecc71"],
            order=LENGTH_ORDER,
            ax=ax,
        )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(MODEL_LABELS[model].replace("\n", " "))
        ax.set_xlabel("Prompt Length")
        if idx == 0:
            ax.set_ylabel("TPOT Change (%)")
        else:
            ax.set_ylabel("")
        ax.set_ylim(-75, 70)
        if idx == 2:
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.get_legend().remove()

    fig.tight_layout()
    return fig


def plot_ttft_scaling(summary: pd.DataFrame) -> plt.Figure:
    """Figure: TTFT scaling with prompt length across platforms.

    Args:
        summary: DataFrame from ``build_summary``.

    Returns:
        Matplotlib Figure with 3 subplots (one per model).
    """
    import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]

    configs = [
        ("gpu", "SDPA+FP16", "GPU: FP16"),
        ("gpu", "SDPA+INT4", "GPU: INT4"),
        ("cpu", "SDPA+FP16", "CPU: FP16"),
        ("rpi", "F16", "RPi: FP16"),
        ("rpi", "Q4KM", "RPi: Q4_K_M"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
    models = list(MODEL_LABELS.keys())

    for idx, model in enumerate(models):
        ax = axes[idx]
        for platform, config, label in configs:
            data = summary[
                (summary["platform"] == platform)
                & (summary["model"] == model)
                & (summary["config"] == config)
            ].sort_values(
                "length",
                key=lambda s: s.map({v: i for i, v in enumerate(LENGTH_ORDER)}),
            )
            if len(data):
                x = [LENGTH_TOKENS[v] for v in data["length"]]
                y = data["ttft_median"].to_numpy()
                ax.plot(x, y, marker="o", label=label, linewidth=1.5, markersize=5)

        ax.set_title(MODEL_LABELS[model].replace("\n", " "))
        ax.set_xlabel("Prompt Tokens")
        if idx == 0:
            ax.set_ylabel("TTFT (ms)")
        ax.set_yscale("log")
        ax.set_xticks([64, 128, 256])
        if idx == 2:
            ax.legend(fontsize=7, loc="center right")

    fig.tight_layout()
    return fig


def generate_figures(
    gpu_dir: Path,
    cpu_dir: Path,
    rpi_dir: Path,
    output_dir: Path,
) -> None:
    """Generate all paper figures and save to *output_dir*.

    Args:
        gpu_dir: Directory with GPU benchmark JSONL files.
        cpu_dir: Directory with CPU benchmark JSONL files.
        rpi_dir: Directory with RPi benchmark JSONL files.
        output_dir: Directory to write PDF/PNG figures to.
    """
    import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
    import pandas as pd
    import seaborn as sns  # ty: ignore[unresolved-import]

    sns.set_theme(style="whitegrid", font_scale=0.95)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_rows = load_results(gpu_dir, "gpu")
    cpu_rows = load_results(cpu_dir, "cpu")
    rpi_rows = load_results(rpi_dir, "rpi")
    df = pd.DataFrame(gpu_rows + cpu_rows + rpi_rows)
    summary = build_summary(df)

    fig1a = plot_tpot_delta(summary)
    fig1a.savefig(output_dir / "fig_tpot_delta.pdf", bbox_inches="tight")
    fig1a.savefig(output_dir / "fig_tpot_delta.png", bbox_inches="tight", dpi=300)
    plt.close(fig1a)

    fig1b = plot_ttft_scaling(summary)
    fig1b.savefig(output_dir / "fig_ttft_scaling.pdf", bbox_inches="tight")
    fig1b.savefig(output_dir / "fig_ttft_scaling.png", bbox_inches="tight", dpi=300)
    plt.close(fig1b)
