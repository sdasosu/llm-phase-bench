# Phase-Wise Analysis of LLM Inference Acceleration on GPU, CPU, and Edge Device

> Published at **[PEARC '26](https://doi.org/10.1145/3785462.3815902)** — Practice and Experience in Advanced Research Computing, July 26–30, 2026, Minneapolis, MN, USA.
> DOI: [10.1145/3785462.3815902](https://doi.org/10.1145/3785462.3815902) · © 2026 the authors, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Overview

LLM Phase Bench is a benchmarking framework that decomposes LLM inference into **prefill** and **decode** phases, enabling fine-grained latency analysis across heterogeneous hardware. We evaluate three sub-1B models on GPU (NVIDIA RTX 3060 Ti), CPU (AMD Ryzen 5 2600), and edge (Raspberry Pi 4B) under varying quantization and attention configurations.

### Key Findings

| | GPU: NF4 | RPi: Q4_K_M | FA2 vs SDPA |
|---|---|---|---|
| **TPOT** | +19–51% (slower) | -55–66% (faster) | &le;5% difference |
| **TTFT** | Scales linearly with prompt length across all platforms |

## Getting Started

**Requirements:** Python ≥ 3.12, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/sdasosu/llm-phase-bench.git
cd llm-phase-bench
uv sync                   # core (GPU / CPU)
uv sync --extra viz       # + plotting (matplotlib, seaborn)
uv sync --extra rpi       # + Raspberry Pi (llama-cpp-python)
```

## Reproducing Results

```bash
# 1. Prepare dataset (SQuAD v1.1, token-length filtered)
benchmark prepare-data --config configs/benchmark/cuda.yaml

# 2. Run benchmark sweep
benchmark run --config configs/benchmark/cuda.yaml
benchmark run --config configs/benchmark/cpu.yaml

# 3. Run on Raspberry Pi (with GGUF models)
benchmark rpi-run --manifest artifacts/data/manifest.json \
                  --models-dir ~/models/gguf

# 4. Aggregate & report
benchmark report --run-dir artifacts/runs/<run_id>
benchmark stats  --gpu-dir artifacts/runs/<gpu> \
                 --cpu-dir artifacts/runs/<cpu> \
                 --rpi-dir artifacts/runs/<rpi>

# 5. Generate figures
benchmark plot   --gpu-dir artifacts/runs/<gpu> \
                 --cpu-dir artifacts/runs/<cpu> \
                 --rpi-dir artifacts/runs/<rpi>
```

## Models

| Model | Parameters | HuggingFace ID |
|-------|-----------|----------------|
| Qwen3.5-0.8B | 0.8B | `Qwen/Qwen3.5-0.8B` |
| Gemma 3 1B-IT | 1.0B | `google/gemma-3-1b-it` |
| Llama 3.2 1B-Instruct | 1.2B | `meta-llama/Llama-3.2-1B-Instruct` |

## Metrics

| Metric | Description |
|--------|-------------|
| TTFT (ms) | Time To First Token &mdash; prefill latency |
| TPOT (ms) | Time Per Output Token &mdash; decode latency |
| EM / F1 | SQuAD-standard Exact Match & Token F1 |

## Project Structure

```
src/llm_phase_bench/
  benchmark/       Inference runner, timing hooks, RPi runner, metrics
  config/          Pydantic experiment schema
  data/            SQuAD loading & prompt formatting
  models/          HuggingFace model/tokenizer loading
  reporting/       Aggregation, plotting, statistics, export
  utils/           Device detection, JSONL I/O
configs/benchmark/ YAML experiment configurations
```

## Citation

If you use this work, please cite our PEARC '26 paper:

```bibtex
@inproceedings{das2026phasewise,
  author    = {Das, Subhransu and Cheng, Jiaming and Vallabhajosyula, Swathi and Soni, Brijesh and Ramnath, Rajiv},
  title     = {Phase-Wise Analysis of {LLM} Inference Acceleration on {GPU}, {CPU}, and Edge Device},
  booktitle = {Practice and Experience in Advanced Research Computing (PEARC '26)},
  year      = {2026},
  location  = {Minneapolis, MN, USA},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3785462.3815902},
  doi       = {10.1145/3785462.3815902},
  isbn      = {979-8-4007-2377-3},
  series    = {PEARC '26},
}
```

