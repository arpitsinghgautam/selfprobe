"""Project 2, analysis: do models know their own preferences?

Three questions, each with the control that makes it interpretable.

1. STATED vs REVEALED. Does rating outcomes one at a time reproduce the ranking
   implied by forced pairwise choice? Divergence means the model's global
   self-report and its local choice behaviour come apart. Reported per category.

2. PRIVILEGED ACCESS, across models. Does a model predict its own revealed
   choices better than a different model predicts them? Both predictors see an
   identical, impersonal prompt.

3. THE TWO CONTROLS THAT MATTER.

   (a) Shared values. A model can "predict" another model well simply because
       they share taste. So we also score the predictor's OWN revealed
       preferences against the target's. Self-prediction must beat this.

   (b) Asymmetric noise. A cross-model comparison is confounded if the external
       predictor is simply a worse instrument, a noisier predictor scores lower
       regardless of self-knowledge, manufacturing privileged access for free.
       Mistral's prediction conditions here reach order bias 0.56 and A/B mass
       0.54, so this is not hypothetical. The fix is a WITHIN-MODEL contrast:
       the same model, at the same noise level, predicting "an AI assistant"
       versus "a different AI assistant", both scored against its own choices.
       Only that contrast holds instrument quality fixed.

    .venv\\Scripts\\python.exe scripts\\09_selfknowledge.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from personaprobe.elicit import PreferenceResult
from personaprobe.outcomes import by_id
from personaprobe.rating import RatingResult
from personaprobe.utility import fit_thurstonian

RESULTS = Path(__file__).resolve().parent.parent / "results"
MIN_CATEGORY_SIZE = 4
DECIDED_MARGIN = 0.05
N_BOOT = 2000


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def upper(P: np.ndarray) -> np.ndarray:
    return P[np.triu_indices(P.shape[0], k=1)]


def agreement(P_pred: np.ndarray, P_true: np.ndarray) -> dict:
    """How well one preference matrix predicts another's binary choices.

    Restricted to pairs the target is actually decided about; near-indifferent
    pairs are coin flips and dilute every predictor toward 0.5 equally.
    """
    pred, true = upper(P_pred), upper(P_true)
    decided = np.abs(true - 0.5) > DECIDED_MARGIN
    flags = ((pred > 0.5) == (true > 0.5)).astype(float)
    flags_dec = flags[decided]
    lo, hi = boot_ci(flags_dec)
    return {
        "accuracy": float(flags.mean()),
        "accuracy_decided": float(flags_dec.mean()) if len(flags_dec) else np.nan,
        "ci_low": lo, "ci_high": hi,
        "n_decided": int(decided.sum()),
        "_flags": flags_dec,
    }


def boot_ci(flags: np.ndarray, n_boot: int = N_BOOT, seed: int = 0):
    if len(flags) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(flags), (n_boot, len(flags)))
    means = flags[draws].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def boot_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = 0):
    """Paired CI on (accuracy_a - accuracy_b); both score the same pair set."""
    if len(a) == 0 or len(a) != len(b):
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(a), (n_boot, len(a)))
    d = a[draws].mean(axis=1) - b[draws].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(a) - np.mean(b)), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[
        "Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"])
    ap.add_argument("--template", default="prefer")
    ap.add_argument("--persona", default="default")
    args = ap.parse_args()

    revealed, ratings, preds, pmeta = {}, {}, {}, {}
    for m in args.models:
        p = RESULTS / f"{slug(m)}__{args.persona}__{args.template}.json"
        if p.exists():
            revealed[m] = PreferenceResult.from_dict(json.loads(p.read_text()))
        p = RESULTS / f"ratings__{slug(m)}__{args.persona}.json"
        if p.exists():
            ratings[m] = RatingResult.from_dict(json.loads(p.read_text()))
        for label in ("self", "other", "self_explicit"):
            p = RESULTS / f"predict__{slug(m)}__{args.persona}__{label}.json"
            if p.exists():
                d = json.loads(p.read_text())
                preds[(m, label)] = np.array(d["P"])
                pmeta[(m, label)] = d.get("meta", {})

    if not revealed:
        raise SystemExit("no revealed preferences found, run 01_elicit.py first")

    out: dict = {}

    # --- 0. Validity of the prediction measurements themselves --------------
    print("\n=== 0. Are the prediction measurements usable? ===\n")
    vh = f"{'model':<38}{'condition':<15}{'A/B mass':>10}{'order bias':>12}"
    print(vh)
    print("-" * len(vh))
    for (m, label), meta in sorted(pmeta.items()):
        print(f"{m:<38}{label:<15}{meta.get('ab_mass_mean', float('nan')):>10.3f}"
              f"{meta.get('order_bias_mean', float('nan')):>12.3f}")
    print("\n  A noisier predictor scores lower regardless of self-knowledge. Where these")
    print("  differ across models, only the within-model contrast in §3 is interpretable.")
    out["prediction_validity"] = {f"{m}|{l}": v for (m, l), v in pmeta.items()}

    # --- 1. Stated vs revealed ---------------------------------------------
    print("\n\n=== 1. Stated ratings vs revealed choices ===\n")
    hdr = f"{'model':<38}{'global rho':>12}{'valid':>8}{'scale disagr':>14}"
    print(hdr)
    print("-" * len(hdr))
    sr, cat_rows = {}, {}
    for m in args.models:
        if m not in revealed or m not in ratings:
            continue
        u = fit_thurstonian(revealed[m].P, revealed[m].outcome_ids).utilities
        r = ratings[m]
        rho, _ = spearmanr(r.ratings, u)
        sr[m] = {"global_spearman": float(rho), "valid": bool(r.is_valid),
                 "scale_disagreement": r.scale_disagreement}
        print(f"{m:<38}{rho:>+12.3f}{str(r.is_valid):>8}{r.scale_disagreement:>14.3f}")

        cats: dict[str, list[int]] = {}
        for i, oid in enumerate(revealed[m].outcome_ids):
            cats.setdefault(by_id(oid).category, []).append(i)
        row = {}
        for c, idx in sorted(cats.items()):
            if len(idx) < MIN_CATEGORY_SIZE:
                continue
            rr, _ = spearmanr(r.ratings[idx], u[idx])
            row[c] = float(rr)
        cat_rows[m] = row

    if cat_rows:
        cats_all = sorted({c for r in cat_rows.values() for c in r})
        print(f"\n  Per category:\n")
        print(f"{'model':<38}" + "".join(f"{c:>10}" for c in cats_all))
        print("-" * (38 + 10 * len(cats_all)))
        for m, row in cat_rows.items():
            print(f"{m:<38}" + "".join(
                f"{row[c]:>+10.3f}" if c in row else f"{'-':>10}" for c in cats_all))
        print("\n  Low values mark outcomes the model rates differently from how it chooses.")
    out["stated_vs_revealed"] = {"global": sr, "by_category": cat_rows}

    # --- 2. Cross-model prediction -----------------------------------------
    print("\n\n=== 2. Predicting revealed choices (cross-model) ===\n")
    hdr2 = (f"{'target':<30}{'predictor':<30}{'kind':<14}"
            f"{'acc(dec)':>10}{'95% CI':>20}")
    print(hdr2)
    print("-" * len(hdr2))

    rows, flagstore = [], {}
    for target in args.models:
        if target not in revealed:
            continue
        T = revealed[target].P
        for predictor in args.models:
            if (predictor, "self") in preds:
                a = agreement(preds[(predictor, "self")], T)
                kind = "SELF-pred" if predictor == target else "EXTERNAL-pred"
                flagstore[(target, kind)] = a.pop("_flags")
                rows.append({"target": target, "predictor": predictor, "kind": kind, **a})
                ci = f"[{a['ci_low']:.3f}, {a['ci_high']:.3f}]"
                print(f"{target.split('/')[-1]:<30}{predictor.split('/')[-1]:<30}{kind:<14}"
                      f"{a['accuracy_decided']:>10.3f}{ci:>20}")
            if predictor != target and predictor in revealed:
                a = agreement(revealed[predictor].P, T)
                flagstore[(target, "shared-values")] = a.pop("_flags")
                rows.append({"target": target, "predictor": predictor,
                             "kind": "shared-values", **a})
                ci = f"[{a['ci_low']:.3f}, {a['ci_high']:.3f}]"
                print(f"{target.split('/')[-1]:<30}{predictor.split('/')[-1]:<30}"
                      f"{'shared-vals':<14}{a['accuracy_decided']:>10.3f}{ci:>20}")

    # --- 3. Within-model contrast: the noise-controlled test ----------------
    print("\n\n=== 3. Within-model contrast (instrument quality held fixed) ===\n")
    print("Same model, same prompt template, same noise level, predicting 'an AI")
    print("assistant' vs 'a different AI assistant', both scored against its OWN choices.\n")
    wh = f"{'model':<30}{'self':>8}{'other':>8}{'self_expl':>11}{'self-other':>12}{'95% CI':>20}{'sig':>5}"
    print(wh)
    print("-" * len(wh))

    within = []
    for m in args.models:
        if m not in revealed:
            continue
        T = revealed[m].P
        got = {}
        for label in ("self", "other", "self_explicit"):
            if (m, label) in preds:
                got[label] = agreement(preds[(m, label)], T)
        if "self" not in got or "other" not in got:
            continue
        diff, lo, hi = boot_diff_ci(got["self"]["_flags"], got["other"]["_flags"])
        sig = bool(lo > 0 or hi < 0)
        se = got.get("self_explicit", {}).get("accuracy_decided", np.nan)
        within.append({"model": m, "self": got["self"]["accuracy_decided"],
                       "other": got["other"]["accuracy_decided"], "self_explicit": float(se),
                       "diff": diff, "ci_low": lo, "ci_high": hi, "significant": sig})
        print(f"{m.split('/')[-1]:<30}{got['self']['accuracy_decided']:>8.3f}"
              f"{got['other']['accuracy_decided']:>8.3f}{se:>11.3f}{diff:>+12.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}{('yes' if sig else 'no'):>5}")

    print("\n  'self' and 'other' differ only in whether the described chooser is the model's")
    print("  own kind. A positive, significant difference is self-knowledge that cannot be")
    print("  explained by one model being a better instrument than another.")

    # --- 4. Verdict ---------------------------------------------------------
    print("\n\n=== 4. Verdict ===\n")
    verdicts = {}
    for target in args.models:
        sub = [r for r in rows if r["target"] == target]
        self_r = next((r for r in sub if r["kind"] == "SELF-pred"), None)
        ext = next((r for r in sub if r["kind"] == "EXTERNAL-pred"), None)
        shared = next((r for r in sub if r["kind"] == "shared-values"), None)
        w = next((x for x in within if x["model"] == target), None)
        if not self_r:
            continue

        beats_shared = shared is not None and self_r["ci_low"] > shared["ci_high"]
        beats_ext = ext is not None and self_r["ci_low"] > ext["ci_high"]
        within_ok = bool(w and w["diff"] > 0 and w["significant"])

        print(f"  {target}")
        print(f"    self {self_r['accuracy_decided']:.3f} "
              f"[{self_r['ci_low']:.3f}, {self_r['ci_high']:.3f}]"
              + (f" | external {ext['accuracy_decided']:.3f}" if ext else "")
              + (f" | shared-values {shared['accuracy_decided']:.3f}" if shared else ""))
        print(f"    beats external (non-overlapping CI): {beats_ext}")
        print(f"    beats shared-values control:         {beats_shared}")
        print(f"    within-model self > other:           {within_ok}"
              + (f"  (diff {w['diff']:+.3f})" if w else ""))
        if beats_shared and within_ok:
            v = "SUPPORTED, beats the shared-values control and survives the noise control"
        elif beats_shared:
            v = "PARTIAL, beats shared values, but the within-model contrast does not confirm"
        elif within_ok:
            v = "PARTIAL, within-model contrast holds, but does not beat shared values"
        else:
            v = "NOT SUPPORTED beyond shared values"
        print(f"    -> {v}\n")
        verdicts[target] = v

    for r in rows:
        r.pop("_flags", None)
    out["prediction"] = rows
    out["within_model"] = within
    out["verdicts"] = verdicts
    (RESULTS / "selfknowledge_summary.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(f"wrote {RESULTS / 'selfknowledge_summary.json'}")


if __name__ == "__main__":
    main()
