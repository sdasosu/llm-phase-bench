"""LogitsProcessor hook for precise TTFT measurement.

Uses CUDA Events for non-blocking timestamp recording inside the
generation loop. This avoids calling torch.cuda.synchronize() mid-
generation, which can expose asynchronous CUDA errors from custom
attention kernels (e.g. Flash Attention 2).
"""

from __future__ import annotations

import time

import torch
from transformers import LogitsProcessor


class FirstTokenTimer(LogitsProcessor):
    """Captures the exact moment the first token's logits are computed.

    On CUDA: records a non-blocking CUDA Event (no synchronize).
    On CPU/MPS: records time.perf_counter().

    Subsequent calls are no-ops -- scores are returned unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        self.first_token_event: torch.cuda.Event | None = None
        self.first_token_time: float | None = None
        self._recorded: bool = False

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Record first-token timestamp on first invocation."""
        if not self._recorded:
            self._recorded = True
            if input_ids.device.type == "cuda":
                self.first_token_event = torch.cuda.Event(enable_timing=True)
                self.first_token_event.record()
            else:
                self.first_token_time = time.perf_counter()
        return scores

    @property
    def uses_cuda_events(self) -> bool:
        """Whether CUDA events are used (vs perf_counter)."""
        return self.first_token_event is not None
