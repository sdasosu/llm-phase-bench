"""QA prompt formatting and chat template application."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def format_qa_prompt(context: str, question: str) -> str:
    """Build a raw QA prompt from context and question.

    This is the *base* prompt before any model-specific chat template is applied.
    """
    return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
) -> str:
    """Wrap a raw prompt with the model-specific chat template.

    Applied at inference time, NOT during data preparation.
    """
    messages = [{"role": "user", "content": prompt}]
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(result, str)
    return result
