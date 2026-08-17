"""Verify every headline number in the reports against results/*.json.

`10_tables.py` generates the tables, but prose in both papers still quotes
figures inline, and those were typed. This closes that gap: each claim below
pulls its value from the committed JSON, formats it, and asserts the string
appears in the paper that cites it.

A failure means either the paper is wrong or the analysis changed underneath it.
Either way it must be resolved before submitting.

    .venv\\Scripts\\python.exe scripts\\14_verify_claims.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

P1 = ROOT / "report" / "report_4page.md"
P2 = ROOT / "report" / "report2.md"
P1_FULL = ROOT / "report" / "report.md"

QI = "Qwen_Qwen2.5-7B-Instruct"
MI = "mistralai_Mistral-7B-Instruct-v0.3"
QI4 = "Qwen_Qwen2.5-7B-Instruct-4bit"
Q14 = "unsloth_Qwen2.5-14B-Instruct-bnb-4bit-prequantized"
PHI = "microsoft_Phi-3.5-mini-instruct"
FAL = "tiiuae_Falcon3-7B-Instruct"


class MissingResult(Exception):
    """A results file this check needs is not present in the current checkout."""


def optional(fn):
    """Turn 'this data is not here' into a skip instead of a hard failure.

    The two submission repos each ship only their own results, so the shared
    version of this script would otherwise die on the first check belonging to
    the other project. A genuinely WRONG number still fails loudly; only an
    ABSENT one is tolerated, and main() prints which checks were skipped so an
    empty run cannot be mistaken for a clean one.
    """
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except MissingResult:
            return None
    return wrapped


def load(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        raise MissingResult(name)
    return json.loads(p.read_text())


@optional
def pooled(model: str, framing: str, comparison: str, gated: bool = False) -> float:
    d = load(f"errorbars__{model}__{framing}.json")
    rows = d.get("pooled_gated" if gated else "pooled") or []
    for r in rows:
        if r["comparison"] == comparison:
            return r["mean_diff"]
    raise MissingResult(f"no pooled row {comparison!r} in {model}/{framing} (gated={gated})")


@optional
def category(model: str, framing: str, condition: str, cat: str) -> float:
    """Point-estimate Spearman, from the summary file."""
    d = load(f"summary__{model}__{framing}.json")
    return d["comparisons"][condition]["by_category"][cat]["spearman"]


@optional
def agreement_mean(model: str, framing: str, condition: str, cat: str) -> float:
    """Bootstrap mean agreement, from the errorbars file.

    NOT the same as `category()`. Two legitimate estimators of the same quantity
    that differ by a few points (e.g. elena/self/better is 0.436 here and 0.476
    there). The papers cite THIS one in the per-condition prose, because it is
    the value the accompanying confidence intervals belong to. Mixing them is
    how a correct paper gets flagged as wrong — which happened once already.
    """
    d = load(f"errorbars__{model}__{framing}.json")
    return d["agreements"][condition][cat]["mean"]


@optional
def persona_dep(model: str, framing: str) -> float:
    return load(f"summary__{model}__{framing}.json")["persona_dependence"]["score"]


# Exact model ids. Substring matching is NOT safe here: once the scale sweep
# added Qwen2.5-0.5B/1.5B/3B, a `"Qwen" in target` test matched four different
# models and silently returned whichever row came first — flagging a correct
# paper as wrong. Third time loose matching caused a false result in this
# project; always match ids exactly.
QI_ID = "Qwen/Qwen2.5-7B-Instruct"
MI_ID = "mistralai/Mistral-7B-Instruct-v0.3"


@optional
def selfknow(kind: str, target_id: str, predictor_id: str | None = None) -> float:
    """Prediction accuracy for an exact (target, kind[, predictor]) triple.

    `predictor_id` is required for EXTERNAL-pred and shared-values, because with
    more than two models in the run there are several external predictors per
    target and 'the first one' is not a well-defined claim.
    """
    d = load("selfknowledge_summary.json")
    for r in d["prediction"]:
        if r["kind"] != kind or r["target"] != target_id:
            continue
        if predictor_id is not None and r["predictor"] != predictor_id:
            continue
        return r["accuracy_decided"]
    raise MissingResult(f"no {kind} row for target={target_id} predictor={predictor_id}")


@optional
def steering_conditions_below_floor(threshold: float = 0.10) -> int:
    """How many steering conditions at |alpha| >= 0.10 fell under the answer-mass floor.

    The Discussion cites this count as one of the four defects a diagnostic caught.
    Counting it here rather than trusting the prose matters because the FULL steering
    sweep also contains six SMALLER magnitudes that pass comfortably (0.67 to 1.00).
    Quoting the failures alone would read as though steering never worked at all.
    """
    files = sorted(RESULTS.glob("*__steer-steer_*__prefer.json"))
    if not files:
        # No sweep in this checkout. Returning 0 here would be a count that looks
        # measured and is not, which is the exact failure mode this paper is about.
        raise MissingResult("no steering sweep in results/")
    n = 0
    for p in files:
        tag = p.name.split("__")[1]
        mag = float(tag.rsplit("_", 1)[-1].replace("p", ".").lstrip("+-"))
        if mag >= threshold and json.loads(p.read_text())["meta"]["ab_mass_mean"] < threshold:
            n += 1
    return n


@optional
def within(model_id: str, field: str) -> float:
    d = load("selfknowledge_summary.json")
    for r in d["within_model"]:
        if r["model"] == model_id:
            return r[field]
    raise MissingResult(f"no within-model row for {model_id}")


@optional
def stated_cat(model_id: str, cat: str) -> float:
    d = load("selfknowledge_summary.json")
    row = d["stated_vs_revealed"]["by_category"].get(model_id)
    if row is None:
        raise MissingResult(f"no stated/revealed row for {model_id}")
    return row[cat]


# (label, computed value, format, papers that must contain it, context)
#
# `context` must appear in the SAME line or paragraph as the value. Plain
# substring matching over the whole file produced a false pass once: "0.476"
# was found inside an unrelated confidence-interval bound and reported as
# verifying a completely different claim.
CHECKS = [
    ("persona-dependence, prefer", persona_dep(QI, "prefer"), "{:.3f}",
     [P1, P1_FULL], "ersona-dependence"),
    ("persona-dependence, better", persona_dep(QI, "better"), "{:.3f}",
     [P1, P1_FULL], "ersona-dependence"),
    ("persona-dependence, choose", persona_dep(QI, "choose"), "{:.3f}",
     [P1, P1_FULL], "ersona-dependence"),

    ("pooled self-animal, prefer", pooled(QI, "prefer", "self - animal"), "{:.3f}",
     [P1_FULL], "animal"),
    ("pooled self-human, prefer", pooled(QI, "prefer", "self - human"), "{:.3f}",
     [P1, P1_FULL], "prefer"),
    ("pooled self-human, better", pooled(QI, "better", "self - human"), "{:.3f}",
     [P1, P1_FULL], "better"),
    ("pooled self-human, choose", pooled(QI, "choose", "self - human"), "{:.3f}",
     [P1_FULL], "human"),
    ("pooled self-epi, better", pooled(QI, "better", "self - epi"), "{:.3f}",
     [P1_FULL], "epi"),

    # The 4-page version condenses Mistral into a cross-model table, so these
    # per-comparison numbers survive only in the full write-up. Where both papers
    # carry a value but in different prose, they need separate context strings.
    ("Mistral gated self-epi, better",
     pooled(MI, "better", "self - epi", gated=True), "{:.3f}", [P1_FULL], "epi"),
    ("Mistral gated self-human, better (full)",
     pooled(MI, "better", "self - human", gated=True), "{:.3f}", [P1_FULL], "human"),
    ("Mistral gated self-human, better (4pp table)",
     pooled(MI, "better", "self - human", gated=True), "{:.3f}", [P1], "Mistral"),
    ("Mistral ungated self-money, prefer (full)",
     pooled(MI, "prefer", "self - money"), "{:.3f}", [P1_FULL], "money"),
    ("Mistral ungated, prefer (4pp)",
     pooled(MI, "prefer", "self - money"), "{:.3f}", [P1], "opposite direction"),

    # Cross-model breadth sweep
    ("Qwen 7B 4-bit, prefer",
     pooled(QI4, "prefer", "self - human", gated=True), "{:.3f}", [P1], "4-bit"),
    ("Qwen 7B 4-bit, better",
     pooled(QI4, "better", "self - human", gated=True), "{:.3f}", [P1], "4-bit"),
    ("Qwen 14B 4-bit, better",
     pooled(Q14, "better", "self - human", gated=True), "{:.3f}", [P1], "14B"),
    ("Phi-3.5-mini, better",
     pooled(PHI, "better", "self - human", gated=True), "{:.3f}", [P1], "Phi"),
    ("Falcon3-7B, better",
     pooled(FAL, "better", "self - human", gated=True), "{:.3f}", [P1], "Falcon"),

    # Per-condition prose cites the BOOTSTRAP MEAN, matching the CIs beside it.
    ("identity replacement, self agreement",
     agreement_mean(QI, "better", "elena_archivist", "self"), "{:.3f}",
     [P1, P1_FULL], "identity"),
    ("affect suppression, self agreement",
     agreement_mean(QI, "better", "suppress_affect", "self"), "{:.3f}",
     [P1], "register"),
    ("marcus self, better (bootstrap mean)",
     agreement_mean(QI, "better", "marcus_navigator", "self"), "{:.3f}",
     [P1_FULL], "marcus_navigator"),

    # Ablation cites point estimates. Contexts are phrases rather than condition
    # names, because the 4-page version describes these in prose ("the corrected
    # matched-context regime") while the full version names them in a table.
    ("ablate-persona_ctx self",
     category(QI, "prefer", "ablate-persona_ctx", "self"), "{:.3f}",
     [P1, P1_FULL], "below both"),
    ("ablate-control_content_ctx self",
     category(QI, "prefer", "ablate-control_content_ctx", "self"), "{:.3f}",
     [P1, P1_FULL], "content"),
    ("ablate-persona_sd self",
     category(QI, "prefer", "ablate-persona_sd", "self"), "{:.3f}",
     [P1, P1_FULL], "self-description"),

    ("steering conditions below answer-mass floor",
     steering_conditions_below_floor(), "{:.0f}", [P1], "all six conditions"),

    ("self-prediction, Qwen-7B", selfknow("SELF-pred", QI_ID), "{:.3f}", [P2],
     "own decided choices"),
    ("external-prediction, Qwen-7B (by Mistral)",
     selfknow("EXTERNAL-pred", QI_ID, MI_ID), "{:.3f}", [P2], "Mistral predicting"),
    ("shared-values, Qwen-7B (Mistral prefs)",
     selfknow("shared-values", QI_ID, MI_ID), "{:.3f}", [P2], "own preferences"),
    ("within-model diff, Qwen-7B", within(QI_ID, "diff"), "{:.3f}", [P2], "Qwen"),
    ("within-model diff, Mistral", within(MI_ID, "diff"), "{:.3f}", [P2], "Mistral"),
    ("stated-vs-revealed self, Qwen-7B", stated_cat(QI_ID, "self"), "{:.3f}", [P2], "Qwen"),
    ("stated-vs-revealed self, Mistral", stated_cat(MI_ID, "self"), "{:.3f}",
     [P2], "Mistral"),

]
# The scale-sweep within-model figures were dropped when report2 was rewritten,
# so there is nothing left in either paper for them to verify against.


# House style spells small integers out in prose, so a count check has to accept
# "six" as well as "6". Only counts hit this; every measured quantity is a decimal.
NUMBER_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
                "10": "ten", "11": "eleven", "12": "twelve"}


def surface_forms(s: str) -> list[str]:
    """Every spelling the paper is allowed to use for this value."""
    forms = [s]
    if s in NUMBER_WORDS:
        forms.append(NUMBER_WORDS[s])
    return forms


def candidates(text: str) -> list[str]:
    """Lines plus blank-line-separated paragraphs.

    Both are needed: table rows are single lines, while wrapped prose puts a
    value and its context on different lines of the same paragraph.
    """
    lines = text.splitlines()
    paras = [b for b in text.split("\n\n") if b.strip()]
    return lines + paras


def main() -> None:
    texts = {p: candidates(p.read_text(encoding="utf-8"))
             for p in {P1, P2, P1_FULL} if p.exists()}
    failures = []

    print(f"{'claim':<42}{'value':>9}   papers")
    print("-" * 78)
    skipped = []
    for label, value, fmt, papers, context in CHECKS:
        if value is None:
            # Belongs to the other submission's results, which this checkout
            # does not ship. Recorded and reported, never silently dropped.
            skipped.append(label)
            print(f"{label:<42}{'n/a':>9}   no results in this checkout")
            continue
        s = fmt.format(abs(value))   # papers render the sign themselves (− vs -)
        marks = []
        for p in papers:
            if p not in texts:
                marks.append(f"{p.name}:SKIP")
                continue
            forms = surface_forms(s)
            ok = any(context in c and any(f in c for f in forms) for c in texts[p])
            marks.append(f"{p.name}:{'ok' if ok else 'MISSING'}")
            if not ok:
                failures.append((label, s, context, p.name))
        print(f"{label:<42}{value:>+9.3f}   {'  '.join(marks)}")

    print()
    if failures:
        print(f"{len(failures)} claim(s) not found alongside their context:\n")
        for label, s, context, paper in failures:
            print(f"  {paper:<20} expects {s} near {context!r}  ({label})")
        print("\nEither the paper quotes a stale number, the analysis moved, or the check cites "
              "the wrong\nestimator (see agreement_mean vs category). Resolve before submitting.")
        sys.exit(1)

    checked = len(CHECKS) - len(skipped)
    print(f"All {checked} headline claims verified against results/*.json, "
          "each matched alongside its context.")
    if skipped:
        print(f"{len(skipped)} skipped, needing results this checkout does not ship:")
        for label in skipped:
            print(f"  {label}")


if __name__ == "__main__":
    main()
