"""SQuAD-standard quality metrics: Normalized Exact Match and Token F1."""

from __future__ import annotations

import re
import string
from typing import TypedDict


class MetricsResult(TypedDict):
    """Result of compute_metrics."""

    mean_em: float
    mean_f1: float
    em_scores: list[float]
    f1_scores: list[float]


def normalize_text(text: str) -> str:
    """Normalize text following SQuAD evaluation conventions.

    Lowercases, removes articles (a, an, the), strips punctuation,
    and collapses whitespace.
    """
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """Binary exact match after normalization."""
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score after normalization.

    Uses whitespace tokenization (standard SQuAD approach, not model
    tokenization).
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_metrics(
    predictions: list[str],
    references: list[list[str]],
) -> MetricsResult:
    """Compute EM and F1 across all samples.

    Takes max score across multiple valid answers per sample (SQuAD convention).

    Args:
        predictions: Model predictions, one per sample.
        references: List of valid reference answers per sample.

    Returns:
        Dict with mean_em, mean_f1, and per-sample lists.
    """
    em_scores: list[float] = []
    f1_scores: list[float] = []

    for pred, refs in zip(predictions, references, strict=True):
        sample_em = max(exact_match(pred, ref) for ref in refs)
        sample_f1 = max(token_f1(pred, ref) for ref in refs)
        em_scores.append(sample_em)
        f1_scores.append(sample_f1)

    n = len(em_scores)
    return {
        "mean_em": sum(em_scores) / n if n > 0 else 0.0,
        "mean_f1": sum(f1_scores) / n if n > 0 else 0.0,
        "em_scores": em_scores,
        "f1_scores": f1_scores,
    }
