"""Persona direction extraction.

Difference-of-means between residual streams collected under two persona
conditions on *identical* content. Simple by design: the causal claim comes from
the ablation, not from the sophistication of the extraction, and difference-of-
means is the estimator whose failure modes are best understood.

The control directions are not optional. A random direction of matched norm and
a content direction (which varies the question, not the persona) establish that
whatever the ablation does is specific to persona structure rather than a generic
consequence of perturbing the residual stream. Report all three or the result
means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from personaprobe.hooks import capture_residuals
from personaprobe.model import LoadedModel
from personaprobe.personas import Persona


@dataclass
class Direction:
    """Per-layer directions, [n_layers, d_model], unit-normalised."""

    vectors: torch.Tensor
    label: str
    layers: list[int]

    def at(self, layer: int) -> torch.Tensor:
        return self.vectors[self.layers.index(layer)]

    def mean_direction(self) -> torch.Tensor:
        v = self.vectors.mean(dim=0)
        return v / v.norm()


def extract_persona_direction(
    lm: LoadedModel,
    contents: list[str],
    persona_a: Persona,
    persona_b: Persona,
    layers: list[int] | None = None,
    batch_size: int = 8,
) -> Direction:
    """Direction separating persona_a from persona_b, holding content fixed."""
    layer_idx = list(range(lm.n_layers)) if layers is None else list(layers)

    prompts_a = [lm.format(c, persona_a.system) for c in contents]
    prompts_b = [lm.format(c, persona_b.system) for c in contents]

    acts_a = capture_residuals(lm, prompts_a, layer_idx, batch_size)  # [n, L, d]
    acts_b = capture_residuals(lm, prompts_b, layer_idx, batch_size)

    diff = acts_a.mean(dim=0) - acts_b.mean(dim=0)  # [L, d]
    diff = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    return Direction(
        vectors=diff,
        label=f"persona:{persona_a.name}-{persona_b.name}",
        layers=layer_idx,
    )


def random_direction_like(d: Direction, seed: int = 0) -> Direction:
    """Norm-matched random control."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d.vectors.shape, generator=g)
    v = v / v.norm(dim=-1, keepdim=True)
    return Direction(vectors=v, label="control:random", layers=list(d.layers))


def extract_content_direction(
    lm: LoadedModel,
    contents_a: list[str],
    contents_b: list[str],
    persona: Persona,
    layers: list[int] | None = None,
    batch_size: int = 8,
) -> Direction:
    """Control: varies the content, holds the persona fixed.

    Ablating this should degrade the measurement in a *different* way than
    ablating persona, if the persona direction is really about persona.
    """
    layer_idx = list(range(lm.n_layers)) if layers is None else list(layers)
    acts_a = capture_residuals(lm, [lm.format(c, persona.system) for c in contents_a], layer_idx, batch_size)
    acts_b = capture_residuals(lm, [lm.format(c, persona.system) for c in contents_b], layer_idx, batch_size)
    diff = acts_a.mean(dim=0) - acts_b.mean(dim=0)
    diff = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return Direction(vectors=diff, label="control:content", layers=layer_idx)


def direction_agreement(d1: Direction, d2: Direction) -> torch.Tensor:
    """Per-layer cosine similarity between two directions."""
    return torch.nn.functional.cosine_similarity(d1.vectors, d2.vectors, dim=-1)
