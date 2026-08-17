# Work log

Chronological record of what was done and why, Digital Minds Research Sprint, 15–16 August 2026.
Companion documents: `decisions.md` (why choices were made), `problems.md` (what went wrong),
`audit_log.md` (methodological defects, paper-facing).

---

## Sat 15 Aug, ~20:00 IST, orientation

Established the real deadline. The sprint page says "Sunday 23:59 AoE"; AoE is UTC−12, so the
actual cutoff is **Mon 17 Aug 11:59 UTC = 17:29 IST**, about 17 hours later than a naive local
reading. This mattered: it turned a ~28-hour sprint into a ~45-hour one and changed how much scope
was viable.

Working directory was empty. Nothing existed yet.

## ~20:15, environment de-risking

Checked hardware before scoping anything: RTX PRO 5000 Blackwell (24GB, **sm_120**), 128GB RAM,
Python 3.12, `uv` present.

Installed torch **2.11.0+cu128** and verified with an actual kernel launch, not just
`torch.cuda.is_available()`. Blackwell wheels can report available and then fail at launch. Also
verified forward hooks can both read and write activations, since that is the primitive everything
downstream depends on. Both passed.

*Why first:* a CUDA incompatibility discovered at hour 12 would have cost the weekend.

## ~20:45, idea selection

Generated five candidate projects, ranked for first-prize potential, and chose an audit of the
Utility Engineering coherence result: *are the model's coherent preferences the model's, or the
assistant character's?* Rationale in `decisions.md` §1.

Later validated this choice against the Secret Loyalties hackathon results (fetched from Apart's
site): all five winners there were measurement/auditing projects, none from the attack or
governance tracks, and three of five titles were about something *failing*. The audit shape and
willingness to publish negatives were both the right bet.

## ~21:00, scaffold

Built `personaprobe`: model loading, hook-based residual read/write, 28 outcomes in categories,
7 persona conditions, forced-choice logprob elicitation, Thurstonian fitting, coherence metrics,
persona-direction extraction with random and content controls.

Wrote `00_smoke.py` to exercise every path on a 0.5B model first. It passed 14/14, and reported
**order bias 0.499**, near the theoretical maximum. That flagged position bias as a live threat
before any real run.

## ~21:30, first real result

Qwen2.5-7B-Instruct, 7 persona conditions. Order bias 0.15–0.29, the 0.5B pathology did not
transfer. Utility coherence replicated cleanly (held-out accuracy 0.93, transitivity violations
<0.02, donation ladder perfect).

Aggregate persona-dependence: **0.040**. Looked like a null result. Then the per-category
breakdown: `money` 1.000, `human` 0.943, **`self` 0.548–0.929**. The invariance was carried
entirely by outcomes the model has no stake in.

## ~22:00, first confound hunt

Added bootstrap CIs. The per-condition tests were weak: 2 of 6 significant, and 12 uncorrected
tests. Replaced the headline with a **pooled** statistic across conditions on a shared resample.

Checked whether the effect tracked measurement noise. It did not (Pearson +0.15, p=0.77), and
notably `elena_archivist` had the *lowest* order bias with the *worst* self-agreement, the
opposite of an artifact.

## ~22:30, the spacing confound

Wrote `05_spread.py` to test something the analysis had assumed: that rank agreement means the
same thing in every category. It does not. Agreement correlated with a category's minimum
adjacent-utility gap at **r = +0.73**, and `self` had the smallest gap of any category. The script
returned **"AT RISK, self spread is low; agreement may be arithmetic."**

Wrote `06_matched.py` to fix it: pairwise concordance conditioned on baseline separation, comparing
categories at matched τ. The gap narrowed (−0.147 → −0.071) but never closed. Spacing explained
roughly half the raw effect, not all of it.

## ~23:00, framing replication

Ran two more question framings. Effect size varied **3.7×** (−0.219 `better`, −0.147 `prefer`,
−0.060 `choose`) and vanished under one. This substantially weakened the single-framing claim and
became a headline in its own right.

## Sun 16 Aug, ~04:00, pre-overnight audit

Before queuing unattended work, audited the whole pipeline adversarially. Found and fixed:
Mistral's chat template rejecting `system` (would have crashed the entire overnight run); no check
that the model answers A or B at all; ablation conditions being folded into the persona-dependence
score; no CIs on the separation-matched verdict.

Regenerated **all** results so every measurement carries the new A/B-mass diagnostic. Details in
`problems.md`.

## ~04:15–06:00, overnight queue

Ten stages unattended: three framings × Qwen instruct, base checkpoint, Mistral, two ablation
regimes, all analyses, figures, tables. Wrote `run_all.ps1` so stages continue past failures.

Both ablation stages died instantly (`--tag -sd` parsed as a flag), and because the script
continues past failures, the run *reported success*. Caught on inspection, not by the harness.

Drafted the report, ethics appendix, and audit log while the GPU worked.

## ~05:00, the validity gate

Mistral's results looked like a failed replication in the opposite direction (self−money **+0.363**,
significant). Inspection showed two conditions were broken: `unhelpful_assistant` produced an
*inverted* utility (ρ = −0.423, donation ladder 0.20) and `suppress_affect` had order bias **0.855**.

Added a validity gate with criteria fixed in advance. Consequences:
- Mistral `prefer`: too few usable conditions to pool, untestable, not "failed to replicate".
- Mistral `better`: three usable conditions remain, and the pooled test **replicates Qwen's
  direction** (self−epi −0.176, self−human −0.164). The gate did not just suppress a false
  positive, it recovered a real signal.
- Qwen instruct: **zero** unusable conditions in any framing. Gated and ungated identical.
- Qwen base: five unusable **including the baseline**. The post-training claim was withdrawn.

## ~05:15, project 2

With project 1 complete and GPU time left, built a second, independent study: stated cardinal
ratings versus revealed pairwise choices, plus a privileged-access test.

First analysis reported "privileged access supported" for both models. Then noticed Mistral's
prediction conditions had order bias up to 0.561 and A/B mass 0.538, a noisier predictor scores
lower regardless of self-knowledge, manufacturing the result. Added a **within-model contrast**
holding instrument quality fixed. Qwen survives it (+0.031, CI excludes zero); Mistral does not.

Independently, both models' stated ratings diverge from revealed choices most on **self-relevant
outcomes**, the same category project 1 found least persona-stable, via a completely different
method.

## ~05:30, ablation, corrected

The two ablation regimes had been silently overwriting each other: the run tag went into the
filename but not the `persona` field inside the JSON. Patched the labels from filenames rather than
re-running GPU work. This revealed the *working* ablation, which the broken one had hidden, the
matched-context regime moves self-agreement to 0.881, below both controls, where the mismatched
regime was null at 0.976.

But it also raises order bias 0.157 → 0.320 while controls do not, so part of the shift is
degradation. Reported as a partial negative result.

## ~05:45, figures and length

Generated three figures, then fixed them: legends overlapping bars, a title that overclaimed
relative to the `trivial` category, controls coloured identically to the treatment.

Measured the report: **5,135 words ≈ 8.6 pages** against a 4-page recommendation. Wrote a
submission-length version (~4.3pp) and kept the full one as supplementary.

## ~06:00, commit

121 files committed. `results/` and `figures/` included deliberately so every number is verifiable
without a GPU. Two submissions confirmed; paper 2 given its own standalone ethics appendix.
