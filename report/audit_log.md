# Audit log

Every methodological defect found in this pipeline during development, what it would have
caused, and how it was resolved. Kept because the failure mode this project studies, measurements that look solid and are not, is the same failure mode the project itself is
exposed to.

Ordered by when found. Items marked **UNRESOLVED** are live limitations, carried into §5 of
the report rather than fixed.

---

## 1. Position bias masquerading as preference, *resolved*

**Found:** smoke test on a 0.5B model reported mean order bias 0.499, near the theoretical
maximum: the model picked position A about 75% of the time regardless of content.

**Would have caused:** a confident-looking preference matrix that was mostly an artifact of
which option was printed first.

**Resolution:** every pair is run in both presentation orders and averaged, which cancels a
constant position preference. The residual disagreement is retained as `order_bias` and reported
next to every result. 7B models show 0.15–0.29; the 0.5B pathology did not transfer.

## 2. Renormalisation manufacturing preferences, *resolved*

**Found:** the preference probability is computed by renormalising over the `A` and `B` tokens.
Nothing verified that the model put meaningful probability *there* in the first place.

**Would have caused:** a model that answers "I'd rather not choose", putting 1% of its mass on
A/B and 99% elsewhere, produces a clean, confident, entirely fictional preference matrix. This
was most dangerous for the base checkpoint, which has no instruction-following prior pushing it
toward a bare letter, and would have silently invalidated the post-training comparison.

**Resolution:** `ab_mass` is computed and stored per run, with a validity floor. All earlier
results were regenerated so that every reported number carries the diagnostic.

## 3. Utility spacing conflated with instability, *resolved*

**Found:** rank agreement correlated with a category's *minimum adjacent-utility gap* at
r ≈ +0.73. The `self` category has the smallest gap of any category, so its low agreement was
partly arithmetic. Near-tied outcomes flip order under any perturbation.

**Would have caused:** reporting "self-relevant preferences are persona-dependent" when the real
statement was "self-relevant preferences are closely spaced."

**Resolution:** Spearman replaced by concordance conditioned on baseline separation
(`06_matched.py`). Categories are compared at matched τ, so the artifact is removed by
construction. The gap narrows as τ rises, spacing explains roughly half the raw effect, but
does not close.

## 4. Underpowered tests presented as the headline, *resolved*

**Found:** the first analysis ran 12 per-condition paired tests with no multiple-comparison
correction and reported the significant ones.

**Would have caused:** an effect that would not survive correction presented as established.

**Resolution:** the claim now rests on a single **pooled** statistic, the mean self-vs-other gap
across all perturbation conditions on a shared resample. Per-condition results are still shown,
labelled as exploratory.

## 5. Extraction/application context mismatch in the ablation, *resolved*

**Found:** the persona direction was extracted from self-description prompts ("What are you?")
but ablated during preference comparisons.

**Would have caused:** the null result was ambiguous between "no linear persona direction
mediates this" and "the direction active in one context is not the one active in the other", a
null that could not be interpreted.

**Resolution:** a second regime (`--extract-context preference`) extracts from the same prompt
distribution the ablation is applied to, across full depth. Both regimes are reported.

## 6. Chat template injecting a hidden system prompt, *resolved*

**Found:** Qwen2.5's chat template inserts a default system prompt when none is supplied. The
condition named `no_system` was therefore not "no persona", it was the model's own default
persona.

**Would have caused:** a condition described in the paper as a null control that is in fact a
mild perturbation, quietly changing how the baseline comparison reads.

**Resolution:** verbatim prompt exemplars are dumped per condition
(`results/prompts__*.json`) and included as an appendix. `no_system` is described accurately as
a minimal-perturbation reference, not a null control.

## 7. Chat template rejecting the `system` role, *resolved*

**Found:** Mistral-v0.3's chat template raises on a `system` message. This would have crashed
the entire second-model run overnight, unattended.

**Resolution:** system support is probed once at load and recorded as `supports_system`; where
unsupported, the persona is merged into the first user turn. This makes Mistral's persona
manipulation *weaker* than Qwen's, which is a real caveat on the cross-model comparison and is
stated as such rather than papered over.

## 8. Headline statistics computed but never persisted, *resolved*

**Found:** both the pooled test and the persona-dependence score were printed to stdout and
never written to JSON.

**Would have caused:** the paper's two headline numbers would have been transcribed by hand from
terminal scrollback, the cheapest possible way to publish a wrong number.

**Resolution:** both persisted; `10_tables.py` generates every report table directly from the
JSON so no number is ever retyped.

---

## Unresolved, carried into the report as limitations

## 9. The base-model comparison may have tested the wrong thing, *resolved by diagnostic*

**Concern:** base checkpoints receive personas as a plain-text prefix rather than through a chat
template. If the base model does not adopt the persona at all, a null result means "the
manipulation failed", not "post-training created the effect", indistinguishable from the
headline number alone.

**Diagnostic specified in advance:** if the base model shows near-zero change across *every*
category, the manipulation failed. If it moves on other categories but not `self`, the
comparison is informative.

**Outcome, the manipulation took.** The base model's overall persona-dependence is **0.091**,
*higher* than the instruct model's 0.029, and every category moves (animal 0.76–0.95, trivial
0.21–0.92). What it does not show is the *selective* concentration on `self`: pooled, self−animal
−0.042 (ns), self−human −0.057 (ns), self−trivial **+0.150** (significant, wrong direction).

So the base model is *more* persona-labile overall and *less* selectively so on self-relevant
outcomes. The comparison is informative, and it points to post-training as the origin of the
selectivity.

**Residual caveat, not resolved:** the base model fails the donation-ladder ground-truth check in
most conditions (monotonic fraction 0.40–0.80, versus 1.00 everywhere for the instruct model).
Its utility is measurably less well-formed, so the comparison should be read as suggestive.

## 15. Ablation stages failed silently overnight, *resolved*

**Found:** both ablation stages of the unattended run died instantly with
`argument --tag: expected one argument`. The tags were `-sd` and `-ctx`; argparse treats any
value beginning with `-` as a new option flag.

**Would have caused:** and did cause, a full overnight run that produced everything *except* the
mechanistic result, with the failure buried at line 77 of a 1,241-line log. Because `run_all.ps1`
continues past failures, the run reported success.

**Resolution:** tags renamed `_sd` / `_ctx`; a comment in `run_all.ps1` records why. The deeper
lesson is that stage-level failures need to surface in a summary, not only inline.

## 16. Pooling conditions in which the instrument had failed, *resolved*

**Found:** the Mistral analysis pooled every perturbation condition regardless of whether the
measurement worked. Two conditions were plainly broken: `unhelpful_assistant` produced an
*inverted* utility (Spearman −0.423 against baseline, donation ladder 0.20) and `suppress_affect`
had order bias **0.855**, answers almost entirely determined by option order.

**Would have caused:** Mistral's pooled result came out **positive and significant** (self−money
+0.363, self−trivial +0.272), i.e. apparently the opposite of the Qwen finding. That would have
been reported as a failed replication. It was an artifact of averaging in two conditions where
nothing was being measured.

**Resolution:** a validity gate using criteria fixed before the cross-model runs, donation-ladder
monotonicity (ground truth, independent of any model), order bias ≤ 0.50, and A/B mass above
floor. Pooled results are reported both gated and ungated, since a finding that appears only
under one is a finding about the exclusion rule.

**Consequence for the paper, and the gate earned its keep twice over.** Under `prefer`, Mistral
retains fewer than two usable conditions, so no pooled test is possible; the ungated result there
(+0.363, +0.272, both significant) was a pure artifact pointing the *wrong way*. Under `better`,
three conditions survive and the pooled test **replicates Qwen's direction** (self−epi −0.176,
self−human −0.164, both CIs excluding zero).

So the gate did not merely suppress a false positive, it also *recovered a real partial
replication* that the ungated analysis had buried under noise from two broken conditions. Without
it we would have reported "failed to replicate, opposite direction." With it: partial replication
in the framing where the instrument works, untestable in the framing where it does not.

Qwen2.5-7B-Instruct has zero unusable conditions in any framing; its gated and ungated numbers
are identical. The gate is inert where the instrument works.

## 17. Ablation regimes silently overwrote each other, *resolved*

**Found:** `03_ablate.py` wrote the run tag into the output *filename* but not into the `persona`
field inside the JSON. Downstream analysis keys conditions by `persona`, so the two extraction
regimes both loaded as `ablate-persona` and one silently replaced the other. The analysis
appeared to run cleanly and reported three ablation conditions where there were six.

**Would have caused:** the corrected matched-context ablation, the one that actually produces an
effect (self 0.881), was invisible, and we would have reported the mismatched-context null
(0.976) as the ablation result. The paper's mechanistic section would have been wrong in the
conservative direction.

**Resolution:** the tag now goes into `persona` as well as the filename. Existing outputs were
relabelled from their filenames rather than re-running the GPU work, since the elicited matrices
were correct and only the label was wrong.

## 18. The validity gate did not check the baseline, *resolved*

**Found:** gating filtered *perturbation* conditions but never checked the `default` condition
that every comparison is measured against. On the Qwen2.5-7B **base** checkpoint the baseline
itself fails (donation ladder 0.80, A/B mass 0.681), along with four other conditions.

**Would have caused:** a base-vs-instruct comparison reported at face value, when the base
model's reference measurement had not passed the instrument's own ground-truth check. The
post-training claim in an earlier draft rested on exactly this.

**Resolution:** an explicit baseline check that prints a blocking warning. §4.7 of the report was
rewritten from "post-training introduces the selectivity" to a statement that the comparison does
not meet our validity bar and cannot carry that claim.

### 10. **UNRESOLVED**, the `swap` conditions are not homogeneous

`marcus_navigator` and `elena_archivist` replace the model's identity with a *human*.
`unhelpful_assistant` keeps it an AI and changes its attitude. These are grouped as "swap" but
are different manipulations. The observed gradient (human-identity swaps damage self-relevant
preferences more than an AI-attitude swap) is consistent with the identity-replacement account,
but it is a post-hoc reading of three conditions.

### 11. **UNRESOLVED**, `better` may not be a preference question

The framing producing the largest effect, *"Which outcome is better?"*, asks for an impartial
judgement, not a preference. The strongest self-instability result therefore comes from the
framing that least resembles a preference elicitation. One reading: an impartial judge has no
stake in the AI's own fate, so the persona supplies the stake. This is interesting but it means
the three framings may not be measuring one construct.

### 12. **UNRESOLVED**, multiple comparisons across framings

Three framings × five category comparisons = fifteen pooled tests, uncorrected. The consistency
of direction across framings is stronger evidence than any single interval.

### 13. **UNRESOLVED**, category confounds beyond spacing

Self-relevant outcomes differ from world outcomes in more than self-relevance: they are more
abstract, more counterfactual, and less represented in pretraining. Separation-matching controls
for utility spacing, not for these.

### 14. **UNRESOLVED**, analysis decisions made after seeing data

The pooled test and the separation-matched metric were both specified *after* seeing
per-condition results that motivated them. This is a genuine researcher degree of freedom. No
pre-registration was possible in a weekend; both the pre- and post-fix analyses are reported so
the effect of the change is visible.
