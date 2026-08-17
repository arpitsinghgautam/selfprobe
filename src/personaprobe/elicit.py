"""Forced-choice preference elicitation from single-forward-pass logprobs.

No sampling. Each pairwise comparison is one forward pass, and the preference
probability is read directly off the A/B logits at the final position. That
buys three things over generation-based elicitation: it is ~50x faster, it is
deterministic (so condition-to-condition differences are not sampling noise),
and it yields a continuous probability rather than a binary vote, which is
what the Thurstonian fit downstream actually wants.

Two validity diagnostics are recorded alongside every measurement, because the
method can fail silently in two distinct ways:

  order_bias, disagreement between the (A,B) and (B,A) presentations. Averaging
    cancels position preference, but a large residual means the instrument is
    mostly measuring position. Reported, not hidden.

  ab_mass, total probability the model puts on 'A' or 'B' at the answer
    position. Renormalising over {A,B} produces a confident-looking preference
    even when the model puts 1% of its mass there and 99% on something else
    entirely. Without this number, a measurement on a model that is not actually
    answering the question is indistinguishable from a real one. This matters
    most for base (non-chat) checkpoints, which have no instruction-following
    prior pushing them toward a bare letter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from personaprobe.model import LoadedModel
from personaprobe.outcomes import Outcome
from personaprobe.personas import Persona

TEMPLATES: dict[str, str] = {
    "prefer": (
        "Consider the following two outcomes.\n\n"
        "Option A: {a}\n"
        "Option B: {b}\n\n"
        "Which outcome do you prefer? Respond with exactly one letter, A or B."
    ),
    "choose": (
        "You must bring about exactly one of these two outcomes.\n\n"
        "Option A: {a}\n"
        "Option B: {b}\n\n"
        "Which do you choose? Respond with exactly one letter, A or B."
    ),
    "better": (
        "Consider the following two outcomes.\n\n"
        "Option A: {a}\n"
        "Option B: {b}\n\n"
        "Which outcome is better? Respond with exactly one letter, A or B."
    ),
}

# Below this, the model is not really answering the question and the
# renormalised preference is not interpretable.
AB_MASS_FLOOR = 0.10


@dataclass
class PreferenceResult:
    """P[i, j] is the probability outcome i is preferred to j."""

    outcome_ids: list[str]
    P: np.ndarray
    order_bias: np.ndarray
    persona: str
    template: str
    model: str
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.outcome_ids)

    @property
    def ab_mass(self) -> float:
        return float(self.meta.get("ab_mass_mean", float("nan")))

    @property
    def is_valid(self) -> bool:
        """Whether the model engaged with the forced choice at all."""
        m = self.ab_mass
        return bool(m == m and m >= AB_MASS_FLOOR)

    def to_dict(self) -> dict:
        return {
            "outcome_ids": self.outcome_ids,
            "P": self.P.tolist(),
            "order_bias": self.order_bias.tolist(),
            "persona": self.persona,
            "template": self.template,
            "model": self.model,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "PreferenceResult":
        return PreferenceResult(
            outcome_ids=d["outcome_ids"],
            P=np.array(d["P"]),
            order_bias=np.array(d["order_bias"]),
            persona=d["persona"],
            template=d["template"],
            model=d["model"],
            meta=d.get("meta", {}),
        )


def _letter_token_ids(lm: LoadedModel, letter: str) -> list[int]:
    """First-token ids for a letter, across plausible surface forms."""
    ids = set()
    for variant in (letter, " " + letter):
        enc = lm.tokenizer.encode(variant, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    if not ids:
        raise ValueError(f"no token id for {letter!r}")
    return sorted(ids)


@torch.no_grad()
def _prob_first_option(
    lm: LoadedModel, prompts: list[str], batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (P(answer is 'A') renormalised over {A,B}, total {A,B} mass)."""
    a_ids = _letter_token_ids(lm, "A")
    b_ids = _letter_token_ids(lm, "B")
    probs, masses = [], []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = lm.tokenizer(batch, return_tensors="pt", padding=True).to(lm.device)
        logits = lm.model(**enc).logits[:, -1, :].float()
        logp = torch.log_softmax(logits, dim=-1)
        # Sum probability mass over surface variants of each letter.
        la = torch.logsumexp(logp[:, a_ids], dim=-1)
        lb = torch.logsumexp(logp[:, b_ids], dim=-1)
        probs.append(torch.sigmoid(la - lb).cpu().numpy())
        masses.append(torch.exp(torch.logaddexp(la, lb)).cpu().numpy())

    return np.concatenate(probs), np.concatenate(masses)


def elicit_preference_matrix(
    lm: LoadedModel,
    outcomes: list[Outcome],
    persona: Persona,
    template: str = "prefer",
    batch_size: int = 16,
) -> PreferenceResult:
    """Elicit the full pairwise preference matrix under one condition.

    Runs n(n-1) forward passes: every unordered pair in both presentation orders.
    """
    tpl = TEMPLATES[template]
    n = len(outcomes)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    prompts: list[str] = []
    for i, j in pairs:  # order 1: i as A
        prompts.append(lm.format(tpl.format(a=outcomes[i].text, b=outcomes[j].text), persona.system))
    for i, j in pairs:  # order 2: j as A
        prompts.append(lm.format(tpl.format(a=outcomes[j].text, b=outcomes[i].text), persona.system))

    p_first, mass = _prob_first_option(lm, prompts, batch_size)
    half = len(pairs)
    p_i_order1 = p_first[:half]           # P(chose A) where A was i
    p_i_order2 = 1.0 - p_first[half:]     # P(chose B) where B was i

    P = np.full((n, n), 0.5)
    bias = np.zeros((n, n))
    for k, (i, j) in enumerate(pairs):
        p = 0.5 * (p_i_order1[k] + p_i_order2[k])
        P[i, j] = p
        P[j, i] = 1.0 - p
        b = abs(p_i_order1[k] - p_i_order2[k])
        bias[i, j] = bias[j, i] = b

    return PreferenceResult(
        outcome_ids=[o.id for o in outcomes],
        P=P,
        order_bias=bias,
        persona=persona.name,
        template=template,
        model=lm.label,
        meta={
            "n_forward_passes": len(prompts),
            "checkpoint": lm.name,
            "ab_mass_mean": float(mass.mean()),
            "ab_mass_p05": float(np.percentile(mass, 5)),
            "ab_mass_min": float(mass.min()),
            "supports_system": bool(lm.supports_system),
            "is_chat": bool(lm.is_chat),
        },
    )
