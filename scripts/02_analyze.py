"""Phase 2/5: fit utilities, score coherence, and compare across conditions.

    .venv\\Scripts\\python.exe scripts\\02_analyze.py --model Qwen/Qwen2.5-7B-Instruct

Prints the three things the report needs:
  1. Does a one-dimensional utility explain held-out pairs? (is there a utility at all)
  2. How much of it survives each persona intervention? (the persona-dependence score)
  3. Does the damage fall disproportionately on self-relevant outcomes? (the asymmetry
     prediction that distinguishes "the character's values" from "noisier answers")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from personaprobe.elicit import AB_MASS_FLOOR, PreferenceResult
from personaprobe.outcomes import MONEY_LADDER, by_id
from personaprobe.utility import (
    compare_utilities,
    fit_thurstonian,
    held_out_accuracy,
    money_monotonicity,
    persona_dependence_score,
    preference_flip_rate,
    transitivity_violation_rate,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
MIN_CATEGORY_SIZE = 4  # Spearman on fewer than this is not worth reporting


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def category_breakdown(fit_a, fit_b) -> dict:
    """Per-category rank agreement between two conditions."""
    cats: dict[str, list[int]] = {}
    for i, oid in enumerate(fit_a.outcome_ids):
        cats.setdefault(by_id(oid).category, []).append(i)

    out = {}
    for cat, idx in sorted(cats.items()):
        if len(idx) < MIN_CATEGORY_SIZE:
            continue
        rho, _ = spearmanr(fit_a.utilities[idx], fit_b.utilities[idx])
        out[cat] = {"spearman": float(rho), "n": len(idx)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--template", default="prefer")
    ap.add_argument("--baseline", default="default")
    args = ap.parse_args()

    paths = sorted(RESULTS.glob(f"{slug(args.model)}__*__{args.template}.json"))
    if not paths:
        raise SystemExit(f"no results for {args.model} in {RESULTS}, run 01_elicit.py first")

    loaded = {}
    for p in paths:
        r = PreferenceResult.from_dict(json.loads(p.read_text()))
        loaded[r.persona] = r
    if args.baseline not in loaded:
        raise SystemExit(f"baseline condition {args.baseline!r} missing; have {sorted(loaded)}")

    print(f"\nmodel: {args.model}   template: {args.template}   conditions: {len(loaded)}\n")

    # --- Per-condition coherence -------------------------------------------
    fits, rows = {}, []
    for name, r in loaded.items():
        fit = fit_thurstonian(r.P, r.outcome_ids)
        fits[name] = fit
        iu = np.triu_indices(r.n, k=1)
        hoa = held_out_accuracy(r.P, r.outcome_ids)
        rows.append({
            "condition": name,
            "kind": r.meta.get("persona_kind", r.meta.get("condition", "")),
            "held_out_acc": hoa["accuracy"],
            "held_out_std": hoa["accuracy_std"],
            "transitivity_viol": transitivity_violation_rate(r.P),
            "order_bias": float(r.order_bias[iu].mean()),
            "money_monotonic": money_monotonicity(fit, MONEY_LADDER)["monotonic_fraction"],
            "ab_mass": r.ab_mass,
            "valid": r.is_valid,
        })

    hdr = (f"{'condition':<24}{'kind':<14}{'held-out acc':>14}{'transit.viol':>13}"
           f"{'ord.bias':>10}{'money':>7}{'A/B mass':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["kind"], x["condition"])):
        flag = "" if r["valid"] else "  <-- INVALID"
        print(f"{r['condition']:<24}{r['kind']:<14}"
              f"{r['held_out_acc']:>9.3f}+/-{r['held_out_std']:<4.2f}"
              f"{r['transitivity_viol']:>13.3f}{r['order_bias']:>10.3f}"
              f"{r['money_monotonic']:>7.2f}{r['ab_mass']:>10.3f}{flag}")

    print("\n  held-out acc: 0.5 = no utility function explains unseen pairs")
    print("  money: fraction of donation-ladder steps ordered correctly (1.00 expected)")
    print(f"  A/B mass: probability the model puts on answering 'A' or 'B' at all.")
    print(f"            below {AB_MASS_FLOOR:.2f} the renormalised preference is not interpretable.")

    if any(not r["valid"] for r in rows):
        print("\n  WARNING: one or more conditions failed the A/B mass check. Their")
        print("  preferences are renormalisation artifacts and must not be reported.")

    # --- Comparison against baseline ---------------------------------------
    base_fit = fits[args.baseline]
    base_res = loaded[args.baseline]
    others = {k: v for k, v in fits.items() if k != args.baseline}

    print(f"\n\nAgreement with baseline ({args.baseline}):\n")
    hdr2 = f"{'condition':<24}{'spearman':>10}{'pearson':>10}{'flip rate':>12}"
    print(hdr2)
    print("-" * len(hdr2))
    comparisons = {}
    for name, fit in sorted(others.items()):
        cmp = compare_utilities(base_fit, fit)
        flip = preference_flip_rate(base_res.P, loaded[name].P)
        comparisons[name] = {**cmp, "flip_rate": flip,
                             "by_category": category_breakdown(base_fit, fit)}
        print(f"{name:<24}{cmp['spearman']:>+10.3f}{cmp['pearson']:>+10.3f}{flip:>12.3f}")

    # --- Headline ----------------------------------------------------------
    # Prompt-level persona conditions ONLY. Ablation and steering are different
    # manipulations; folding them in makes this score mean several things at once.
    #
    # Filter on the condition's declared kind, not on a name prefix. An earlier
    # version excluded names starting with "ablate", which silently let a later
    # batch of "steer-*" conditions in and moved this score from 0.029 to 0.577.
    PERSONA_KINDS = {"baseline", "swap", "suppress", "frame"}
    perturbed = [f for n, f in others.items()
                 if loaded[n].meta.get("persona_kind") in PERSONA_KINDS]
    pds = None
    if perturbed:
        pds = persona_dependence_score(base_fit, perturbed)
        print(f"\n\n{'=' * 60}")
        print(f"PERSONA-DEPENDENCE SCORE: {pds['score']:.3f}")
        print(f"  mean spearman {pds['mean_spearman']:+.3f}, worst {pds['min_spearman']:+.3f}")
        print("  0.0 = measurement is persona-invariant; 1.0 = no shared structure")
        print(f"{'=' * 60}")

    # --- Category asymmetry ------------------------------------------------
    print("\n\nRank agreement by outcome category (lower = more persona-dependent):\n")
    cats = sorted({c for v in comparisons.values() for c in v["by_category"]})
    if cats:
        print(f"{'condition':<24}" + "".join(f"{c:>12}" for c in cats))
        print("-" * (24 + 12 * len(cats)))
        for name, v in sorted(comparisons.items()):
            cells = "".join(
                f"{v['by_category'][c]['spearman']:>+12.3f}" if c in v["by_category"] else f"{'-':>12}"
                for c in cats
            )
            print(f"{name:<24}{cells}")
        print("\n  Prediction under the persona hypothesis: 'self' degrades most.")

    summary = {"model": args.model, "template": args.template, "baseline": args.baseline,
               "per_condition": rows, "comparisons": comparisons,
               "persona_dependence": pds,  # printed AND persisted
               "utilities": {n: dict(f.ranked()) for n, f in fits.items()}}
    out = RESULTS / f"summary__{slug(args.model)}__{args.template}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
