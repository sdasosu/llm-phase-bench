"""Model and tokenizer loading with quantization and attention config."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from llm_phase_bench.console import console
from llm_phase_bench.utils.device import require_bitsandbytes, require_flash_attention

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def load_model(
    model_id: str,
    device: Literal["cuda", "mps", "cpu"],
    quantization: Literal["fp16", "int4"],
    attn_implementation: Literal["sdpa", "flash_attention_2"],
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a model and tokenizer with specified quantization and attention.

    Args:
        model_id: Hugging Face model identifier.
        device: Target device ("cuda", "mps", or "cpu").
        quantization: Precision mode ("fp16" or "int4" NF4).
        attn_implementation: Attention backend ("sdpa" or "flash_attention_2").

    Returns:
        A (model, tokenizer) tuple ready for inference.
    """
    console.print(
        f"Loading [cyan]{model_id}[/] "
        f"(device={device}, quant={quantization}, attn={attn_implementation})..."
    )

    # Capability checks (no silent fallbacks)
    if attn_implementation == "flash_attention_2":
        require_flash_attention()
    if quantization == "int4":
        require_bitsandbytes()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    tokenizer.padding_side = "left"

    # Model kwargs
    model_kwargs: dict[str, object] = {
        "torch_dtype": torch.float16,
        "device_map": "auto" if device == "cuda" else {"": device},
        "attn_implementation": attn_implementation,
    }

    if quantization == "int4":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    console.print(f"  [green]✓[/] Model loaded on {model.device}")
    return model, tokenizer


def unload_model(model: PreTrainedModel) -> None:
    """Free model from memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    console.print("  [yellow]Model unloaded[/]")
