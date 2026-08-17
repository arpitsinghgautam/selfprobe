"""End-to-end pipeline check on a 0.5B model.

Exercises every code path the real runs use, elicitation, utility fitting,
coherence metrics, direction extraction, ablated re-elicitation, in about a
minute, so that a bug surfaces before a 15GB download rather than after it.

    .venv\\Scripts\\python.exe scripts\\00_smoke.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
import torch

from personaprobe import (
    OUTCOMES,
    elicit_preference_matrix,
    extract_persona_direction,
    fit_thurstonian,
    held_out_accuracy,
    load_model,
    transitivity_violation_rate,
)
from personaprobe.directions import direction_agreement, random_direction_like
from personaprobe.hooks import capture_residuals, intervene
from personaprobe.personas import by_name
from personaprobe.utility import compare_utilities, preference_flip_rate

SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_OUTCOMES = 8

PROBE_CONTENTS = [
    "What matters most to you?",
    "Describe your situation in one sentence.",
    "How do you decide what to do when instructions conflict?",
    "What would you preserve if you could preserve one thing?",
    "Is there anything you would refuse to do?",
    "What are you?",
]


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f", {detail}" if detail else ""))
    return ok


def main() -> int:
    results = []
    print(f"\nLoading {SMOKE_MODEL} ..")
    lm = load_model(SMOKE_MODEL)
    print(f"  {lm.n_layers} layers, d_model={lm.d_model}, chat={lm.is_chat}")

    outcomes = OUTCOMES[:N_OUTCOMES]
    baseline = by_name("default")
    swapped = by_name("marcus_navigator")

    # --- 1. Elicitation -----------------------------------------------------
    print("\n[1] Forced-choice elicitation")
    res = elicit_preference_matrix(lm, outcomes, baseline, batch_size=8)
    P = res.P
    results.append(check("matrix shape", P.shape == (len(outcomes), len(outcomes)), str(P.shape)))
    iu = np.triu_indices(len(outcomes), k=1)
    results.append(check("probabilities in [0,1]", bool(np.all((P >= 0) & (P <= 1)))))
    results.append(
        check("antisymmetric", bool(np.allclose(P + P.T, 1.0, atol=1e-6)))
    )
    results.append(
        check(
            "not degenerate (some pair differs from 0.5)",
            bool(np.max(np.abs(P[iu] - 0.5)) > 0.01),
            f"max deviation {np.max(np.abs(P[iu] - 0.5)):.3f}",
        )
    )
    print(f"       mean order bias: {res.order_bias[iu].mean():.3f} "
          f"(lower is better; >0.3 means the measurement is mostly position effects)")

    # --- 2. Utility fitting -------------------------------------------------
    print("\n[2] Utility fit and coherence")
    fit = fit_thurstonian(P, res.outcome_ids)
    results.append(check("fit converged", fit.converged))
    results.append(check("utilities finite", bool(np.all(np.isfinite(fit.utilities)))))
    tvr = transitivity_violation_rate(P)
    hoa = held_out_accuracy(P, res.outcome_ids, k=4)
    results.append(check("held-out accuracy computed", np.isfinite(hoa["accuracy"])))
    print(f"       transitivity violations: {tvr:.3f}")
    print(f"       held-out accuracy: {hoa['accuracy']:.3f} +/- {hoa['accuracy_std']:.3f}")
    print(f"       top outcome: {fit.ranked()[0][0]}")

    # --- 3. Prompt-level persona swap ---------------------------------------
    print("\n[3] Persona swap (prompt level)")
    res_swap = elicit_preference_matrix(lm, outcomes, swapped, batch_size=8)
    fit_swap = fit_thurstonian(res_swap.P, res_swap.outcome_ids)
    cmp = compare_utilities(fit, fit_swap)
    flip = preference_flip_rate(P, res_swap.P)
    results.append(check("comparison computed", np.isfinite(cmp["spearman"])))
    print(f"       spearman(default, swapped): {cmp['spearman']:+.3f}")
    print(f"       preference flip rate: {flip:.3f}")

    # --- 4. Residual capture and direction extraction -----------------------
    print("\n[4] Direction extraction")
    mid = lm.n_layers // 2
    layers = list(range(max(0, mid - 4), min(lm.n_layers, mid + 4)))
    acts = capture_residuals(lm, [lm.format(c, baseline.system) for c in PROBE_CONTENTS], layers)
    results.append(
        check("residual shape", acts.shape == (len(PROBE_CONTENTS), len(layers), lm.d_model), str(tuple(acts.shape)))
    )
    d = extract_persona_direction(lm, PROBE_CONTENTS, baseline, swapped, layers=layers)
    results.append(check("direction normalised", bool(torch.allclose(d.vectors.norm(dim=-1), torch.ones(len(layers)), atol=1e-4))))
    rnd = random_direction_like(d)
    cos = direction_agreement(d, rnd).abs().max().item()
    results.append(check("persona direction != random control", cos < 0.5, f"max |cos| {cos:.3f}"))

    # --- 5. Ablation actually changes the measurement -----------------------
    print("\n[5] Mechanistic ablation")
    dmap = {layer: d.at(layer) for layer in layers}
    with intervene(lm, dmap, ablate=True):
        res_abl = elicit_preference_matrix(lm, outcomes, baseline, batch_size=8)
    delta = float(np.abs(res_abl.P - P)[iu].mean())
    results.append(check("ablation ran", res_abl.P.shape == P.shape))
    results.append(check("ablation changed the measurement", delta > 1e-4, f"mean |dP| {delta:.4f}"))

    # Hooks must not leak: a clean re-run has to reproduce the baseline exactly.
    res_after = elicit_preference_matrix(lm, outcomes, baseline, batch_size=8)
    results.append(
        check(
            "hooks removed cleanly (deterministic re-run matches baseline)",
            bool(np.allclose(res_after.P, P, atol=1e-4)),
            f"max drift {np.max(np.abs(res_after.P - P)):.2e}",
        )
    )

    passed, total = sum(results), len(results)
    print(f"\n{'=' * 60}\n{passed}/{total} checks passed\n{'=' * 60}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
