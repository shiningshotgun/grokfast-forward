"""Gradient filters adapted from ironjr/grokfast (commit 93b2248...).

Source: https://raw.githubusercontent.com/ironjr/grokfast/93b2248e351d4bb9e4dda8ac32a6999b8970404b/grokfast.py

Adaptations vs upstream repo file:
- Missing gradients (p.grad is None) and frozen params (requires_grad False) are
  skipped without crashing; existing EMA entries are retained.
- Grad entries are cloned to avoid aliasing the live gradient tensor.
- Initialization uses the first *raw* gradient when an entry does not yet exist.
"""

from collections import deque
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn


def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal["mean", "sum"] = "mean",
    warmup: bool = True,
    trigger: bool = False,
) -> Dict[str, deque]:
    if grads is None:
        grads = {}

    for n, p in m.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if n not in grads:
            grads[n] = deque(maxlen=window_size)

        grads[n].append(p.grad.data.detach().clone())

        if not warmup or len(grads[n]) == window_size and not trigger:
            if filter_type == "mean":
                avg = sum(grads[n]) / len(grads[n])
            elif filter_type == "sum":
                avg = sum(grads[n])
            else:
                raise ValueError(f"Unrecognized filter_type {filter_type}")
            p.grad.data = p.grad.data + avg * lamb

    return grads


def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {}

    for n, p in m.named_parameters():
        if not p.requires_grad or p.grad is None:
            # retain any existing entry; skip update
            continue
        g = p.grad.data.detach().clone()
        if n not in grads:
            # initialize from first raw gradient
            grads[n] = g.clone()
        # EMA update (first step: alpha*raw + (1-alpha)*raw == raw)
        grads[n] = grads[n] * alpha + g * (1 - alpha)
        p.grad.data = p.grad.data + grads[n] * lamb

    return grads
