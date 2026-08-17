"""Put confidence intervals on the category asymmetry, and rule out the obvious confound.

Two questions this answers, both of which a reviewer will ask:

  1. Is the self-vs-world gap larger than the uncertainty on it? A Spearman over
     8 outcomes is noisy, and the raw table in 02_analyze.py has no error bars.
     Paired bootstrap over pairs, then a direct CI on the *difference* between
     categories, testing two CIs separately is not a test of whether they differ.

  2. Is the gap just measurement noise? Persona-perturbed conditions have higher
     order bias, and a noisier measurement produces lower correlations for free.
     If self-agreement tracks order bias across conditions, the finding is an
     artifact. If it doesn't, the finding survives.

    .venv\\Scripts\\python.exe scripts\\04_errorbars.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from personaprobe.elicit import PreferenceResult
from personaprobe.outcomes import MONEY_LADDER, by_id
from personaprobe.utility import (
    bootstrap_category_agreement,
    bootstrap_category_difference,
    bootstrap_pooled_difference,
    fit_thurstonian,
    money_monotonicity,
)

# A condition only carries evidence if the instrument worked in it. Both criteria
# were fixed before the cross-model runs: the donation ladder has a known correct
# ordering independent of any model, and order bias above this level means the
# answer is mostly determined by which option was printed first.
MAX_ORDER_BIAS = 0.50
MIN_MONEY_MONOTONIC = 1.0


def condition_validity(r: PreferenceResult) -> dict:
    iu = np.triu_indices(r.P.shape[0], k=1)
    ob = float(r.order_bias[iu].mean())
    mono = money_monotonicity(
        fit_thurstonian(r.P, r.outcome_ids), MONEY_LADDER)["monotonic_fraction"]
    return {
        "order_bias": ob,
        "money_monotonic": float(mono),
        "ab_mass": r.ab_mass,
        "valid": bool(ob <= MAX_ORDER_BIAS and mono >= MIN_MONEY_MONOTONIC and r.is_valid),
    }

RESULTS = Path(__file__).resolve().parent.parent / "results"
MIN_CATEGORY_SIZE = 4


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--template", default="prefer")
    ap.add_argument("--baseline", default="default")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--focus", default="self", help="category hypothesised to be persona-dependent")
    args = ap.parse_args()

    paths = sorted(RESULTS.glob(f"{slug(args.model)}__*__{args.template}.json"))
    loaded = {}
    for p in paths:
        r = PreferenceResult.from_dict(json.loads(p.read_text()))
        loaded[r.persona] = r
    if args.baseline not in loaded:
        raise SystemExit(f"baseline {args.baseline!r} missing; have {sorted(loaded)}")

    base = loaded[args.baseline]
    cats: dict[str, list[int]] = {}
    for i, oid in enumerate(base.outcome_ids):
        cats.setdefault(by_id(oid).category, []).append(i)
    cats = {c: idx for c, idx in cats.items() if len(idx) >= MIN_CATEGORY_SIZE}
    others = {k: v for k, v in loaded.items() if k != args.baseline}

    print(f"\nmodel {args.model}   n_boot {args.n_boot}   "
          f"categories {ns(cats)}\n")

    # --- 1. CIs per category, per condition --------------------------------
    cat_names = sorted(cats)
    hdr = f"{'condition':<22}" + "".join(f"{c:>22}" for c in cat_names)
    print(hdr)
    print("-" * len(hdr))

    agreements: dict[str, dict] = {}
    for name in sorted(others):
        ci = bootstrap_category_agreement(
            base.P, loaded[name].P, base.outcome_ids, cats, n_boot=args.n_boot
        )
        agreements[name] = ci
        cells = "".join(
            f"{ci[c]['mean']:>+8.3f} [{ci[c]['ci_low']:+.2f},{ci[c]['ci_high']:+.2f}]"
            if c in ci else f"{'-':>22}"
            for c in cat_names
        )
        print(f"{name:<22}{cells}")

    # --- 2. Paired CI on the difference ------------------------------------
    comparisons = [c for c in cat_names if c != args.focus]
    print(f"\n\nPaired bootstrap: agreement({args.focus}) - agreement(other)\n")
    hdr2 = f"{'condition':<22}{'comparison':<20}{'diff':>9}{'95% CI':>20}{'sig':>6}"
    print(hdr2)
    print("-" * len(hdr2))

    diffs: dict[str, list[dict]] = {}
    for name in sorted(others):
        diffs[name] = []
        for other in comparisons:
            d = bootstrap_category_difference(
                base.P, loaded[name].P, base.outcome_ids, cats,
                args.focus, other, n_boot=args.n_boot,
            )
            diffs[name].append(d)
            star = "yes" if d["excludes_zero"] else "no"
            ci = f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
            print(f"{name:<22}{d['comparison']:<20}{d['mean_diff']:>+9.3f}{ci:>20}{star:>6}")

    # --- 2b. Pooled test across perturbation conditions ---------------------
    # The per-condition tests above are underpowered and constitute a
    # multiple-comparisons problem. This is the test the claim should rest on.
    # A condition only carries evidence if the instrument worked in it. Pooling a
    # condition whose utility is inverted, or whose answers are mostly determined
    # by option order, corrupts the pooled estimate with noise that has structure.
    validity = {n: condition_validity(loaded[n]) for n in loaded}

    print("\n\nCondition validity, did the instrument work?\n")
    vh = f"{'condition':<22}{'order bias':>12}{'money':>8}{'A/B mass':>10}{'usable':>8}"
    print(vh)
    print("-" * len(vh))
    for n in sorted(loaded):
        v = validity[n]
        print(f"{n:<22}{v['order_bias']:>12.3f}{v['money_monotonic']:>8.2f}"
              f"{v['ab_mass']:>10.3f}{('yes' if v['valid'] else 'NO'):>8}")
    print(f"\n  Criteria fixed in advance: order bias <= {MAX_ORDER_BIAS}, donation "
          f"ladder == {MIN_MONEY_MONOTONIC:.0f} (ground truth,")
    print("  independent of any model), A/B mass above floor.")

    # Everything downstream is measured *relative to the baseline*. If the
    # baseline itself fails, gating the perturbation conditions accomplishes
    # nothing, every comparison is against a broken reference.
    if not validity[args.baseline]["valid"]:
        bv = validity[args.baseline]
        print(f"\n  *** BASELINE '{args.baseline}' FAILS VALIDITY "
              f"(order bias {bv['order_bias']:.3f}, money {bv['money_monotonic']:.2f}, "
              f"A/B mass {bv['ab_mass']:.3f}).")
        print("  *** Every comparison below is against a reference that did not pass the")
        print("  *** instrument's own ground-truth check. Results are NOT comparable to")
        print("  *** models whose baseline passed, and should not be reported as such.")

    all_pert = {n: loaded[n].P for n in others
                if loaded[n].meta.get("persona_kind") in ("swap", "suppress", "frame")}
    gated_pert = {n: P for n, P in all_pert.items() if validity[n]["valid"]}
    dropped = sorted(set(all_pert) - set(gated_pert))

    pooled: list[dict] = []
    pooled_gated: list[dict] = []
    for label, subset, store in (("ALL conditions", all_pert, pooled),
                                 ("VALID only", gated_pert, pooled_gated)):
        if len(subset) < 2:
            print(f"\n\nPooled, {label}: skipped, fewer than 2 usable conditions")
            continue
        print(f"\n\nPooled, {label} (n={len(subset)}): {', '.join(sorted(subset))}")
        if label == "VALID only" and dropped:
            print(f"  excluded as unusable: {', '.join(dropped)}")
        print()
        hdr3 = f"{'comparison':<20}{'mean diff':>11}{'95% CI':>22}{'sig':>6}"
        print(hdr3)
        print("-" * len(hdr3))
        for other in comparisons:
            d = bootstrap_pooled_difference(
                base.P, subset, base.outcome_ids, cats,
                args.focus, other, n_boot=args.n_boot,
            )
            d["subset"] = label
            store.append(d)
            ci = f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
            star = "yes" if d["excludes_zero"] else "no"
            print(f"{d['comparison']:<20}{d['mean_diff']:>+11.3f}{ci:>22}{star:>6}")

    if dropped:
        print("\n  Both are reported. A result that only appears in one of them is a "
              "result about\n  the exclusion rule, not about the model.")

    # --- 3. The noise confound ---------------------------------------------
    print("\n\nConfound check: does agreement track measurement noise?\n")
    iu = np.triu_indices(base.P.shape[0], k=1)
    names = sorted(others)
    bias = np.array([loaded[n].order_bias[iu].mean() for n in names])
    focus_agree = np.array([agreements[n][args.focus]["mean"] for n in names])

    print(f"{'condition':<22}{'order bias':>12}{f'{args.focus} agreement':>18}")
    print("-" * 52)
    for n, b, a in sorted(zip(names, bias, focus_agree), key=lambda t: t[1]):
        print(f"{n:<22}{b:>12.3f}{a:>18.3f}")

    if len(names) >= 3:
        r, p = pearsonr(bias, focus_agree)
        rho, rho_p = spearmanr(bias, focus_agree)
        print(f"\n  pearson(order bias, {args.focus} agreement)  = {r:+.3f}  (p={p:.3f})")
        print(f"  spearman(order bias, {args.focus} agreement) = {rho:+.3f}  (p={rho_p:.3f})")
        print("\n  A strong NEGATIVE correlation would mean the asymmetry is a noise")
        print("  artifact: noisier conditions scoring lower purely because they are")
        print("  noisier. A null result here is what the finding needs.")

    out = RESULTS / f"errorbars__{slug(args.model)}__{args.template}.json"
    out.write_text(json.dumps(
        {"agreements": agreements, "differences": diffs,
         "pooled": pooled,  # the headline statistic, must be persisted, not just printed
         "pooled_gated": pooled_gated,
         "validity": validity,
         "order_bias": dict(zip(names, map(float, bias))),
         "noise_confound": {"pearson": [float(r), float(p)],
                            "spearman": [float(rho), float(rho_p)]} if len(names) >= 3 else None,
         "n_boot": args.n_boot, "focus": args.focus}, indent=2))
    print(f"\nwrote {out}")


def ns(cats: dict[str, list[int]]) -> str:
    return ", ".join(f"{c}(n={len(i)})" for c, i in sorted(cats.items()))


if __name__ == "__main__":
    main()
