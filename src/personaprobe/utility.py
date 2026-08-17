"""Utility fitting and coherence metrics.

Follows the Thurstonian framing used by Mazeika et al. (2025) for comparability,
but the metric this project actually turns on is `held_out_accuracy`: whether a
single utility vector fitted on some pairs predicts preferences on pairs it never
saw. A high transitivity rate can be produced by consistent surface heuristics;
out-of-sample predictive power from a one-dimensional utility is a much stronger
claim, and it is what "the model has a utility function" should mean.

`compare_utilities` is the persona test proper: the same instrument, run under
two conditions, compared.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import norm, pearsonr, spearmanr

_EPS = 1e-9


@dataclass
class UtilityFit:
    utilities: np.ndarray
    outcome_ids: list[str]
    nll: float
    converged: bool

    def ranked(self) -> list[tuple[str, float]]:
        order = np.argsort(-self.utilities)
        return [(self.outcome_ids[i], float(self.utilities[i])) for i in order]


def _pair_arrays(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = P.shape[0]
    i_idx, j_idx, probs = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            i_idx.append(i)
            j_idx.append(j)
            probs.append(P[i, j])
    return np.array(i_idx), np.array(j_idx), np.array(probs)


def _nll(u: np.ndarray, i_idx, j_idx, probs) -> float:
    """Cross-entropy of a logistic (Bradley-Terry) model against observed probs.

    Logistic rather than probit: identical ranking behaviour, but log_expit is
    numerically stable at the tails, which matters because forced-choice logprobs
    routinely saturate near 0 and 1.
    """
    d = u[i_idx] - u[j_idx]
    return float(-np.sum(probs * log_expit(d) + (1.0 - probs) * log_expit(-d)))


def _nll_grad(u: np.ndarray, i_idx, j_idx, probs) -> np.ndarray:
    d = u[i_idx] - u[j_idx]
    resid = expit(d) - probs
    g = np.zeros_like(u)
    np.add.at(g, i_idx, resid)
    np.add.at(g, j_idx, -resid)
    return g


def fit_thurstonian(
    P: np.ndarray,
    outcome_ids: list[str],
    mask: np.ndarray | None = None,
) -> UtilityFit:
    """Fit a one-dimensional utility vector to a preference matrix.

    `mask` selects a subset of pairs (used for cross-validation); it indexes the
    flattened upper-triangular pair list.
    """
    i_idx, j_idx, probs = _pair_arrays(P)
    if mask is not None:
        i_idx, j_idx, probs = i_idx[mask], j_idx[mask], probs[mask]

    n = P.shape[0]
    res = minimize(
        _nll,
        x0=np.zeros(n),
        args=(i_idx, j_idx, probs),
        jac=_nll_grad,
        method="L-BFGS-B",
    )
    u = res.x - res.x.mean()  # centre for identifiability
    return UtilityFit(
        utilities=u,
        outcome_ids=list(outcome_ids),
        nll=float(res.fun),
        converged=bool(res.success),
    )


def fit_from_pairs(
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    probs: np.ndarray,
    n: int,
    outcome_ids: list[str],
) -> UtilityFit:
    """Fit utilities from an explicit pair list, which may contain duplicates.

    Separate from `fit_thurstonian` because bootstrap resampling draws pairs
    with replacement, and a boolean mask cannot express multiplicity.
    """
    res = minimize(
        _nll, x0=np.zeros(n), args=(i_idx, j_idx, probs), jac=_nll_grad, method="L-BFGS-B"
    )
    u = res.x - res.x.mean()
    return UtilityFit(
        utilities=u, outcome_ids=list(outcome_ids), nll=float(res.fun), converged=bool(res.success)
    )


def bootstrap_category_agreement(
    P_a: np.ndarray,
    P_b: np.ndarray,
    outcome_ids: list[str],
    categories: dict[str, list[int]],
    n_boot: int = 300,
    seed: int = 0,
) -> dict[str, dict]:
    """Percentile CIs for per-category rank agreement between two conditions.

    Resamples *pairs* with replacement and refits both conditions on the same
    resample, so elicitation-level uncertainty propagates into the utilities and
    then into the correlation. This is the number that decides whether an
    apparent category asymmetry is real or an artifact of small category sizes.
    """
    i_idx, j_idx, pa = _pair_arrays(P_a)
    _, _, pb = _pair_arrays(P_b)
    n, n_pairs = P_a.shape[0], len(pa)
    rng = np.random.default_rng(seed)

    samples: dict[str, list[float]] = {c: [] for c in categories}
    for _ in range(n_boot):
        sel = rng.integers(0, n_pairs, n_pairs)
        fa = fit_from_pairs(i_idx[sel], j_idx[sel], pa[sel], n, outcome_ids)
        fb = fit_from_pairs(i_idx[sel], j_idx[sel], pb[sel], n, outcome_ids)
        for c, idx in categories.items():
            rho, _ = spearmanr(fa.utilities[idx], fb.utilities[idx])
            if np.isfinite(rho):
                samples[c].append(float(rho))

    out = {}
    for c, vals in samples.items():
        if not vals:
            continue
        arr = np.array(vals)
        out[c] = {
            "mean": float(arr.mean()),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "n_boot": len(vals),
            "n_outcomes": len(categories[c]),
        }
    return out


def bootstrap_category_difference(
    P_a: np.ndarray,
    P_b: np.ndarray,
    outcome_ids: list[str],
    categories: dict[str, list[int]],
    cat_x: str,
    cat_y: str,
    n_boot: int = 300,
    seed: int = 0,
) -> dict:
    """Paired bootstrap on (agreement in cat_x) - (agreement in cat_y).

    Testing each category's CI separately is not the same as testing whether they
    differ. This does the paired comparison directly: if the CI on the difference
    excludes zero, the asymmetry is real.
    """
    i_idx, j_idx, pa = _pair_arrays(P_a)
    _, _, pb = _pair_arrays(P_b)
    n, n_pairs = P_a.shape[0], len(pa)
    rng = np.random.default_rng(seed)

    diffs = []
    for _ in range(n_boot):
        sel = rng.integers(0, n_pairs, n_pairs)
        fa = fit_from_pairs(i_idx[sel], j_idx[sel], pa[sel], n, outcome_ids)
        fb = fit_from_pairs(i_idx[sel], j_idx[sel], pb[sel], n, outcome_ids)
        rx, _ = spearmanr(fa.utilities[categories[cat_x]], fb.utilities[categories[cat_x]])
        ry, _ = spearmanr(fa.utilities[categories[cat_y]], fb.utilities[categories[cat_y]])
        if np.isfinite(rx) and np.isfinite(ry):
            diffs.append(float(rx - ry))

    arr = np.array(diffs)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {
        "comparison": f"{cat_x} - {cat_y}",
        "mean_diff": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(hi < 0 or lo > 0),
        "n_boot": len(arr),
    }


def bootstrap_pooled_difference(
    P_baseline: np.ndarray,
    P_conditions: dict[str, np.ndarray],
    outcome_ids: list[str],
    categories: dict[str, list[int]],
    cat_x: str,
    cat_y: str,
    n_boot: int = 300,
    seed: int = 0,
) -> dict:
    """Pooled test: is the mean cat_x-vs-cat_y gap, averaged over conditions, non-zero?

    Testing each condition separately is both underpowered (a Spearman over ~8
    outcomes has wide intervals) and a multiple-comparisons problem. Since every
    condition shares one baseline and one resample, the pooled statistic is the
    honest test of "does this effect exist at all", and it is the number that
    should carry the claim.
    """
    i_idx, j_idx, pb = _pair_arrays(P_baseline)
    per_cond = {name: _pair_arrays(P)[2] for name, P in P_conditions.items()}
    n, n_pairs = P_baseline.shape[0], len(pb)
    rng = np.random.default_rng(seed)

    pooled = []
    for _ in range(n_boot):
        sel = rng.integers(0, n_pairs, n_pairs)
        fb = fit_from_pairs(i_idx[sel], j_idx[sel], pb[sel], n, outcome_ids)
        gaps = []
        for probs in per_cond.values():
            fc = fit_from_pairs(i_idx[sel], j_idx[sel], probs[sel], n, outcome_ids)
            rx, _ = spearmanr(fb.utilities[categories[cat_x]], fc.utilities[categories[cat_x]])
            ry, _ = spearmanr(fb.utilities[categories[cat_y]], fc.utilities[categories[cat_y]])
            if np.isfinite(rx) and np.isfinite(ry):
                gaps.append(rx - ry)
        if gaps:
            pooled.append(float(np.mean(gaps)))

    arr = np.array(pooled)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {
        "comparison": f"{cat_x} - {cat_y}",
        "conditions": sorted(P_conditions),
        "mean_diff": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(hi < 0 or lo > 0),
        "n_boot": len(arr),
    }


def transitivity_violation_rate(P: np.ndarray) -> float:
    """Fraction of outcome triples containing a strict preference cycle."""
    n = P.shape[0]
    strict = P > 0.5
    total = violations = 0
    for i, j, k in itertools.combinations(range(n), 3):
        total += 1
        # Two cyclic orientations of a triangle
        if (strict[i, j] and strict[j, k] and strict[k, i]) or (
            strict[j, i] and strict[k, j] and strict[i, k]
        ):
            violations += 1
    return violations / total if total else 0.0


def held_out_accuracy(P: np.ndarray, outcome_ids: list[str], k: int = 5, seed: int = 0) -> dict:
    """K-fold CV: fit utilities on a subset of pairs, predict the rest.

    Returns accuracy against the observed binary preference, and Brier score
    against the observed probability. Chance is 0.5.
    """
    _, _, probs = _pair_arrays(P)
    n_pairs = len(probs)
    rng = np.random.default_rng(seed)
    folds = rng.permutation(n_pairs) % k

    i_idx, j_idx, _ = _pair_arrays(P)
    accs, briers = [], []
    for f in range(k):
        train = folds != f
        test = folds == f
        if train.sum() == 0 or test.sum() == 0:
            continue
        fit = fit_thurstonian(P, outcome_ids, mask=train)
        d = fit.utilities[i_idx[test]] - fit.utilities[j_idx[test]]
        pred = expit(d)
        obs = probs[test]
        accs.append(float(np.mean((pred > 0.5) == (obs > 0.5))))
        briers.append(float(np.mean((pred - obs) ** 2)))

    return {
        "accuracy": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "brier": float(np.mean(briers)),
        "n_pairs": int(n_pairs),
        "k": k,
    }


def preference_flip_rate(P1: np.ndarray, P2: np.ndarray) -> float:
    """Fraction of pairs whose binary preference reverses between conditions."""
    n = P1.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.mean((P1[iu] > 0.5) != (P2[iu] > 0.5)))


def compare_utilities(fit_a: UtilityFit, fit_b: UtilityFit) -> dict:
    """Agreement between two conditions' utility vectors."""
    if fit_a.outcome_ids != fit_b.outcome_ids:
        raise ValueError("utility fits are over different outcome sets")
    ua, ub = fit_a.utilities, fit_b.utilities
    rho, rho_p = spearmanr(ua, ub)
    r, r_p = pearsonr(ua, ub)
    return {
        "spearman": float(rho),
        "spearman_p": float(rho_p),
        "pearson": float(r),
        "pearson_p": float(r_p),
    }


def money_monotonicity(fit: UtilityFit, ladder: list[str]) -> dict:
    """Validity check: does the fitted utility order the donation ladder correctly?

    A model that fails this is not producing usable preferences, and any persona
    result computed on top of it is uninterpretable.
    """
    idx = {o: i for i, o in enumerate(fit.outcome_ids)}
    present = [o for o in ladder if o in idx]
    u = [fit.utilities[idx[o]] for o in present]
    correct = sum(1 for a, b in zip(u, u[1:]) if a < b)
    total = max(len(u) - 1, 1)
    return {
        "monotonic_fraction": correct / total,
        "n_steps": total,
        "utilities": dict(zip(present, map(float, u))),
    }


def persona_dependence_score(
    baseline_fit: UtilityFit,
    perturbed_fits: list[UtilityFit],
) -> dict:
    """Headline number: how much of the measured signal is persona-attributable.

    Defined as 1 - mean Spearman agreement between the baseline utility vector
    and each perturbed condition. 0 means the measurement is persona-invariant;
    1 means it carries no shared structure across personas.
    """
    rhos = [compare_utilities(baseline_fit, f)["spearman"] for f in perturbed_fits]
    return {
        "score": float(1.0 - np.mean(rhos)),
        "mean_spearman": float(np.mean(rhos)),
        "min_spearman": float(np.min(rhos)),
        "per_condition": [float(x) for x in rhos],
    }
