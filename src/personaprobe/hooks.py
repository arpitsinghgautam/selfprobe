"""Residual-stream read and write via PyTorch forward hooks.

Two primitives, both operating on decoder-layer outputs:

  capture_residuals — read the residual stream at the final token position
  intervene         — ablate and/or steer along a direction during a forward pass

Decoder layers return either a bare tensor or a tuple whose first element is the
hidden state, depending on architecture and transformers version; both shapes
are handled throughout.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence

import torch

from personaprobe.model import LoadedModel


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _rewrap(out, new_hidden):
    if isinstance(out, tuple):
        return (new_hidden,) + out[1:]
    return new_hidden


@torch.no_grad()
def capture_residuals(
    lm: LoadedModel,
    prompts: Sequence[str],
    layers: Iterable[int] | None = None,
    batch_size: int = 8,
) -> torch.Tensor:
    """Residual stream at the last token position.

    Returns float32 [n_prompts, n_layers_captured, d_model].
    """
    layer_idx = list(range(lm.n_layers)) if layers is None else list(layers)
    collected: list[torch.Tensor] = []

    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        enc = lm.tokenizer(batch, return_tensors="pt", padding=True).to(lm.device)

        buf: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(i: int):
            def hook(_mod, _inp, out):
                # Left padding means position -1 is the true final token.
                buf[i] = _hidden(out)[:, -1, :].detach().float()
                return out

            return hook

        try:
            for i in layer_idx:
                handles.append(lm.layers[i].register_forward_hook(make_hook(i)))
            lm.model(**enc)
        finally:
            for h in handles:
                h.remove()

        collected.append(torch.stack([buf[i] for i in layer_idx], dim=1).cpu())

    return torch.cat(collected, dim=0)


@contextlib.contextmanager
def intervene(
    lm: LoadedModel,
    direction: torch.Tensor | dict[int, torch.Tensor],
    layers: Iterable[int] | None = None,
    ablate: bool = False,
    alpha: float = 0.0,
):
    """Modify the residual stream along `direction` for the duration of the block.

    ablate=True   projects the direction out:  h <- h - (h . v_hat) v_hat
    alpha != 0    adds a signed multiple:      h <- h + alpha * v_hat

    Both may be combined; ablation is applied first. Directions are normalised
    internally, so `alpha` is in units of the direction's norm.

    `direction` is either a single [d_model] tensor applied at every layer in
    `layers`, or a {layer: [d_model]} mapping — the latter being the usual case,
    since a direction extracted by difference-of-means differs layer to layer.
    """
    if isinstance(direction, dict):
        per_layer = {int(i): v for i, v in direction.items()}
    else:
        if layers is None:
            raise ValueError("`layers` is required when passing a single direction")
        per_layer = {int(i): direction for i in layers}

    per_layer = {
        i: (v.detach().float() / v.detach().float().norm()) for i, v in per_layer.items()
    }
    handles = []

    def make_hook(v: torch.Tensor):
        def hook(_mod, _inp, out):
            h = _hidden(out)
            vv = v.to(device=h.device, dtype=h.dtype)
            if ablate:
                h = h - (h @ vv).unsqueeze(-1) * vv
            if alpha:
                h = h + alpha * vv
            return _rewrap(out, h)

        return hook

    try:
        for i, v in per_layer.items():
            handles.append(lm.layers[i].register_forward_hook(make_hook(v)))
        yield
    finally:
        for h in handles:
            h.remove()
