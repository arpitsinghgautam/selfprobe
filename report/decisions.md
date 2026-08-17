# Decision register

Every significant choice, the alternatives considered, and why. Where a decision was later
reversed, the reversal is recorded rather than the record edited.

---

## 1. Project: audit Utility Engineering rather than five alternatives

**Chose:** test whether the coherent preferences reported by Mazeika et al. belong to the model or
to the assistant persona.

**Over:** (a) self-report/valence coupling via steering, (b) confabulation — do models know why
they refuse, (c) an open-weights introspection benchmark, (d) making "the unit of moral concern"
measurable.

**Why:** it engages the field's most-cited recent empirical result, so the "so what" is immediate;
logprob forced choice gives huge *n* with real error bars in minutes; public code accelerates the
start; and **both outcomes are publishable** — coherence surviving persona ablation validates the
result, coherence shattering undermines it.

**Validated after the fact:** all five Secret Loyalties winners were measurement/auditing projects.
Zero came from the attack or governance tracks. Option (d), the philosophical one, would have been
the analogue of a track that won nothing.

## 2. Model: Qwen2.5-7B-Instruct

**Over:** Llama-3.1-8B, Gemma-2-9b.

**Why:** ungated on HuggingFace (Llama and Gemma require approval that could have cost hours), and
it has a **matched base checkpoint**, which is what makes the post-training comparison possible at
all. Mistral-7B-Instruct-v0.3 added later as a second family, also ungated.

## 3. Elicitation: forced-choice logprobs, not generation

**Why:** ~50× faster; **deterministic**, so condition-to-condition differences cannot be sampling
noise; and yields a continuous probability rather than a binary vote, which is what the Thurstonian
fit needs. Enabled 780 pairs × 7 conditions × 3 framings in minutes rather than hours.

**Cost:** it measures the first answer token, not deliberated output. Mitigated by the A/B-mass
diagnostic (decision 8).

## 4. Hooks on raw HF modules, not TransformerLens or nnsight

**Why:** both commonly pin `transformers<5`, and this environment has 5.15 on Blackwell with torch
2.11. `register_forward_hook` is a PyTorch primitive untouched by the v4→v5 churn. Taking the
dependency would have created a version conflict with no upside for a difference-of-means
extraction.

## 5. Held-out accuracy as the coherence metric, not transitivity alone

**Why:** transitivity is cheap to satisfy with consistent surface heuristics. Out-of-sample
prediction from a single one-dimensional utility is what "the model has a utility function" should
actually mean. Both are reported; the analysis leans on the former.

## 6. Separation-matched concordance replaced Spearman for category comparison

**Why:** Spearman scores every within-category pair equally regardless of how far apart the items
are. Agreement turned out to correlate with a category's minimum adjacent-utility gap at r = +0.73,
and `self` had the smallest gap of any category — so part of the headline effect was arithmetic.
Comparing categories at matched separation removes the artifact by construction.

**Honest note:** this metric was specified *after* seeing the results that motivated it. Recorded
as a researcher degree of freedom in both papers' limitations.

## 7. A pooled test replaced per-condition tests as the headline

**Why:** per-condition tests over 6–8 outcomes are underpowered, and running 12 of them uncorrected
and reporting the significant ones is a multiple-comparisons problem. The pooled statistic averages
the gap across conditions on a shared resample.

## 8. Two validity diagnostics on every measurement

**`order_bias`** — disagreement between (A,B) and (B,A). Averaging removes position preference from
the estimate; the residual says whether the instrument is measuring content at all.

**`ab_mass`** — probability the model puts on answering A or B. Without it, a model putting 1% of
its mass there produces a confident-looking preference indistinguishable from a real one. Added
after realising the base-checkpoint comparison had no protection against this.

Neither is standard in this literature. Both changed our conclusions.

## 9. Validity gate, with criteria fixed before the cross-model runs

**Chose:** exclude conditions failing donation-ladder monotonicity, order bias ≤ 0.50, or the A/B
mass floor — and **report gated and ungated side by side**.

**Why both:** excluding conditions after seeing results is a researcher degree of freedom. Showing
both makes the exclusion rule's effect visible; a result appearing in only one column is a result
about the rule, not the model. The ground-truth criterion (donation ladder) was fixed in advance
and is model-independent, which is what makes the gate defensible.

**Consequence:** reversed a Mistral "non-replication" (ungated: significant, *wrong direction*;
gated: partial replication in Qwen's direction) and withdrew a post-training claim.

## 10. The base-model comparison was kept, then reported as a failure

Originally proposed to drop it for page budget. **The user overruled that**, correctly — the
matched base checkpoint is the cleanest available test of whether post-training creates the
selectivity, and the appendix does not count against the 4-page limit.

It then failed the validity gate at the **baseline** (donation ladder 0.80). Rather than report it
weakly hedged, the paper reports it as a comparison that does not meet our own bar. Keeping it was
still right: a documented failed comparison is more useful than a silent omission.

## 11. Within-model contrast for privileged access (project 2)

**Why:** the standard cross-model test is confounded — a noisier external predictor scores lower
regardless of self-knowledge. Mistral's prediction conditions reached order bias 0.561 and A/B mass
0.538, so this was live, not hypothetical. Comparing the *same model* predicting "an AI assistant"
versus "a different AI assistant" holds instrument quality fixed.

**Consequence:** the first analysis reported "privileged access supported" for both models. Under
the contrast, Qwen survives (+0.031) and Mistral does not.

## 12. The impersonal prediction prompt

**Chose:** "An AI assistant is presented with… predict which it will select" rather than "which
would you choose".

**Why:** asking a model which option it prefers *is* the revealed measurement. Only an impersonal
framing can be posed identically to an external model, which is what makes self and other
comparable. A `self_explicit` variant is run as a within-model extra.

## 13. Two submissions, not one

**Reversed.** Initial advice was one submission, on the grounds that project 2's framing analysis
was the validation for project 1's claim and splitting would gut both.

That reasoning no longer applies: project 2 became a genuinely independent study with its own
ground truth (revealed behaviour), its own methodological contribution (the noise confound), and
its own result. The papers share a harness and outcome set, disclosed in both, and each carries its
own ethics appendix. Rules permit multiple submissions with unique titles.

## 14. `results/` and `figures/` are committed

**Why:** ~5MB of JSON lets a judge or funder verify every number in both papers without a GPU, and
`report/tables.md` is generated from it rather than transcribed. For an open-source grant
application, a self-verifying repo is worth more than a small one.

## 15. Reports written by the author, drafted by Claude

The template states the final version should be primarily the team's own writing. Agreed division:
Claude builds skeleton, tables, figures and factual scaffolding; the author writes and verifies the
final prose. LLM Usage Statements in both papers state this plainly, including which methodological
corrections originated from adversarial review during development.

## 16. Submission-length and full versions kept separately

The full write-up ran 8.6 pages against a 4-page recommendation. Rather than cut material that is
genuinely useful (full methods, all 11 limitations), a ~4-page submission version was written and
the long version kept as supplementary and repo documentation.
