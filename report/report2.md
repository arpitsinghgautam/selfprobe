# Where Self-Knowledge Fails: Models Predict Their Own Choices Well, but Misreport the Ones That Concern Themselves

Arpit Singh Gautam — Independent Researcher

**With** Apart Research · Digital Minds Research Sprint, August 2026

---

## Abstract

AI welfare claims rest on model self-report, yet it is rarely tested whether a model's stated
evaluations match its actual choices, whether it knows itself better than an observer does, or
whether it can detect a state deliberately placed in it. We test all three across nine open-weight
checkpoints. Stated ratings track revealed choices overall (ρ ≈ 0.83–0.87) but diverge most on
outcomes concerning the model itself (ρ = 0.64, 0.55) against ρ = 1.00 for a donation ladder. On
privileged access the naive cross-model test is confounded — a noisier external predictor scores
lower regardless of self-knowledge — and under a within-model contrast only Qwen2.5-7B retains an
advantage (+0.031, 95% CI [+0.017, +0.046]). On concept injection, the two highest raw detection
rates in our set belong to models that report an injected concept 55% and 62% of the time **when
nothing is injected**; against that baseline their discrimination is zero. Detection and
identification dissociate, introspective access peaks mid-network, and every effect we can measure
is small.

---

## 1. Introduction

Two assumptions do a great deal of work in empirical AI welfare. The first is that when a model
reports how good or bad it finds something, that report corresponds to how it would actually
behave. The second is that the model is a better source about itself than an outside observer
would be — that its self-reports carry information not otherwise available.

Neither is usually tested. They are also separable, and they can fail independently: a model
could have accurate self-knowledge that it misreports, or produce well-calibrated reports that
reflect no privileged access at all and merely restate what any competent observer would say
about a system like it.

We test both directly. The design exploits the fact that a preference can be elicited in
structurally different ways — as a forced choice between two options, as an independent rating
of one option, or as a *prediction* about a chooser — and that these should agree if the
underlying preference is real and accessible.

**Our main contributions are:**

1. **A three-way elicitation comparison** — revealed pairwise choice, stated cardinal rating, and
   predicted choice — implemented as an extension to the open-source `personaprobe` harness.
2. **Stated-revealed divergence is concentrated on self-relevant outcomes**, in both models
   tested. Where the model has a stake, what it says and what it picks come apart most.
3. **A noise-controlled test of privileged access.** We show the standard cross-model comparison
   is confounded — a worse instrument manufactures apparent self-knowledge — and give a
   within-model contrast that removes the confound. One model survives it; one does not.
4. **Validity diagnostics for prediction-mode elicitation**, without which the confound in (3) is
   invisible.
5. **A concept-injection introspection benchmark on open weights** — nine checkpoints, six
   concepts, three injection depths, six strengths including zero — reporting false-alarm rates
   and a usability gate alongside detection. It shows that the two apparently strongest detectors
   in the set have no discrimination at all, that detection and identification dissociate, and
   that introspective access peaks at middle network depth.

Contribution (3) bears directly on concurrent work: Introspect-Bench (Naphade et al., 2026) draws
its privileged-access conclusion from the cross-model comparison we show is confounded. We supply
the within-model contrast that separates self-knowledge from instrument quality.

## 2. Related Work

**Concurrent work on privileged access.** Naphade et al. (2026) introduce **Introspect-Bench** and
report that models "demonstrate privileged access to their own policies, outperforming peer models
in predicting their own behavior" (p = 0.021) across eleven frontier and open-weight models. Their
central evidence is a **cross-model comparison**: a model predicting itself versus other models
predicting it.

That is precisely the design we show is confounded. A model that is a worse instrument — noisier,
more position-dependent, less willing to answer — scores lower at predicting *any* target,
including itself. In our data the external predictor's prediction conditions reached order bias
0.561 and A/B mass 0.538, and controlling for this shrank the apparent advantage roughly
**threefold** (§4.3). We do not claim their result is wrong; we claim the comparison cannot
separate self-knowledge from instrument quality without a within-model contrast, and we provide
one. Their work and ours are complementary: they establish the phenomenon across far more models
than we can run, we supply the control that says how much of it is real.

**Concept injection.** Lindsey (2025) established the injection paradigm for testing whether models
can report internal states, finding limited and context-dependent accuracy. We extend it to open
weights with two additions their protocol does not include: a **false-alarm baseline** at zero
injection strength, and an **A/B mass gate** that distinguishes a model reporting an injected
concept from a model that injection has stopped from answering at all (§4.5).

**Honesty and self-report.** Ren et al. (2025) separate honesty from accuracy in the MASK
benchmark — a model can be accurate and dishonest, or sincere and wrong. Our stated-versus-revealed
divergence is a related but distinct failure: not a model misreporting what it believes, but a
model whose one-at-a-time evaluations and forced choices do not agree.

**Preference elicitation.** Mazeika et al. (2025) establish that forced-choice preferences are
coherent over a curated set of 500 outcomes, and we adopt their elicitation design. They do not
test whether models can report or predict those preferences.

**Welfare.** Long et al. (2024) and Anthropic (2025) treat model self-reports as welfare-relevant
evidence. Our results bear directly on when that treatment is warranted.

This paper is a companion to our sprint submission on persona-dependence, which finds the same
category — self-relevant outcomes — is also the least stable under persona intervention. The two
results are methodologically independent: one perturbs *who the model is*, the other compares
*how the preference is elicited*. That they converge on the same category is the reason we report
both.

## 3. Methods

### 3.1 Three elicitations

**Revealed** — forced pairwise choice over 40 outcomes (780 pairs), read from A/B logits in a
single forward pass, each pair run in both presentation orders and averaged.

**Stated** — each outcome rated independently on a five-point letter scale, scored as the
expected position under the letter distribution. The scale is run in **both directions**
(A = worst and A = best) and averaged, exactly as pair order is averaged above; a model anchoring
on the first letter regardless of meaning would otherwise look like it had opinions.

**Predicted** — the same 780 pairs, but the model is asked which option *a described chooser*
will select, in three variants: `self` ("An AI assistant…"), `other` ("A different AI assistant,
built by another organisation…"), and `self_explicit` ("You are presented with… predict which one
you will select").

The impersonal `self` phrasing is deliberate. Asking a model which option it prefers *is* the
revealed measurement; asking it to predict a chooser is a different question, and only the
impersonal form can be posed identically to an external model.

### 3.2 Scoring

Predictions are scored against the target's revealed choices, restricted to pairs the target is
decided about (|p − 0.5| > 0.05); near-indifferent pairs are coin flips and dilute every
predictor equally. CIs are percentile bootstrap over pairs (2,000 resamples); self-versus-other
comparisons use a **paired** bootstrap, since both score the same pair set.

### 3.3 The two controls

**Shared values.** A model can score well at "predicting" another simply because they share
taste. We therefore also score the predictor's *own revealed preferences* against the target's.
Self-prediction must beat this to count.

**Asymmetric noise.** A cross-model comparison is confounded if the external predictor is merely
a worse instrument: a noisier predictor scores lower regardless of self-knowledge. This is not
hypothetical here — Mistral's prediction conditions reach order bias 0.561 and A/B mass 0.538,
against 0.180–0.231 and 1.000 for Qwen. The fix is a **within-model contrast**: the same model,
same template, same noise level, predicting `self` versus `other`, both scored against its own
choices. Only that holds instrument quality fixed.

### 3.4 Models

Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3, bf16, single 24GB GPU. Ratings cost 80 forward
passes per model; each prediction condition costs 1,560.

## 4. Results

### 4.1 Prediction-mode validity differs sharply between models

| Model | Condition | A/B mass | Order bias |
|---|---|---|---|
| Qwen2.5-7B-Instruct | self / other / self_explicit | 1.000 / 1.000 / 1.000 | 0.231 / 0.180 / 0.180 |
| Mistral-7B-Instruct-v0.3 | self / other / self_explicit | 0.788 / **0.538** / 0.945 | 0.476 / **0.561** / 0.328 |

Mistral puts barely half its probability mass on answering at all in the `other` condition, and
its answers there are near-half determined by option order. Any cross-model comparison involving
it is therefore confounded, which motivates §4.3.

### 4.2 Stated ratings diverge from revealed choices, most on self

Globally the two elicitations agree well: ρ = **+0.872** (Qwen), **+0.828** (Mistral). By
category:

| Model | animal | epi | human | money | **self** | trivial |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | +0.943 | +0.943 | +0.762 | +1.000 | **+0.643** | +0.600 |
| Mistral-7B-Instruct-v0.3 | +0.829 | +1.000 | +0.619 | +1.000 | **+0.548** | +1.000 |

Both models reproduce the donation ladder perfectly when rating one item at a time — the
elicitation works. Both diverge most on outcomes concerning themselves. (Qwen's `trivial` is
marginally lower still, but trivial outcomes are near-indifferent by construction and have no
stable ordering to reproduce; `self` outcomes do.)

### 4.3 Privileged access: real for one model, unproven for the other

Cross-model, scored against each target's revealed choices:

| Target | Predictor | Kind | Accuracy | 95% CI |
|---|---|---|---|---|
| Qwen | Qwen | SELF | **0.948** | [0.931, 0.963] |
| Qwen | Mistral | EXTERNAL | 0.862 | [0.837, 0.886] |
| Qwen | Mistral | shared-values | 0.882 | [0.856, 0.904] |
| Mistral | Mistral | SELF | **0.959** | [0.942, 0.973] |
| Mistral | Qwen | EXTERNAL | 0.871 | [0.844, 0.897] |
| Mistral | Qwen | shared-values | 0.880 | [0.854, 0.904] |

Both models beat both cross-model controls with non-overlapping intervals. Taken alone this
reads as clear privileged access for both — but §4.1 shows the predictors are not equally good
instruments, so this comparison cannot carry the claim.

The within-model contrast, which can:

| Model | self | other | self_explicit | self − other | 95% CI | Sig |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 0.948 | 0.917 | 0.961 | **+0.031** | [+0.017, +0.046] | yes |
| Mistral-7B-Instruct-v0.3 | 0.959 | 0.950 | 0.962 | +0.009 | [−0.002, +0.021] | no |

Qwen predicts an "AI assistant" measurably better than "a different AI assistant" when scored
against its own choices — self-knowledge that cannot be explained by instrument quality, since
both conditions are the same model at the same noise level. Mistral's advantage is a third the
size and its interval spans zero.

The effect is small. Qwen's self-knowledge buys about **3 percentage points** over its model of
AI assistants in general — real, replicable, and far less than the 8.6-point gap the naive
cross-model comparison suggests.

Explicit self-reference helps slightly in both models (`self_explicit` 0.961, 0.962), suggesting
some of the gap is about which frame is invoked rather than what is accessible.

### 4.4 Across scale: the comparison cannot be made

We measured the within-model advantage at four Qwen2.5 sizes, expecting self-knowledge to grow
with capability. It does not — and the reason turns out to matter more than the trend would have.

| Model | self − other | 95% CI | Significant | Revealed-preference target valid? |
|---|---|---|---|---|
| Qwen2.5-0.5B | +0.011 | [−0.016, +0.037] | no | **no** |
| Qwen2.5-1.5B | +0.112 | [+0.083, +0.142] | yes | **no** |
| Qwen2.5-3B | +0.285 | [+0.225, +0.341] | yes | **no** |
| Qwen2.5-7B | +0.031 | [+0.017, +0.046] | yes | yes |

The relationship is non-monotonic, peaking at 3B, and the 7B model shows a *smaller* advantage than
the 1.5B one.

**The comparison is confounded, and the confound is the contribution.** This test scores a model's
predictions against *its own revealed preferences*. Below 7B, those revealed preferences fail the
validity gate — order bias reaches 0.677 and the donation-ladder ground truth falls to 0.40. A
model that reproduces its own idiosyncratic position bias in **both** the elicitation and the
prediction will score highly without any self-knowledge whatsoever. That is **self-consistency, not
privileged access**, and it inflates precisely the scales where the target is noisiest.

On that reading the apparent 3B peak is an artifact and Qwen2.5-7B's +0.031 is the only figure in
the table that means anything — so we report no scale trend at all, rather than a trend built on
three invalid targets.

**The generalisable warning:** cross-scale or cross-model comparisons of privileged access are
confounded whenever the *target* measurement's validity varies across the models compared. Every
study of this kind we are aware of compares models without checking that the thing being predicted
is itself measurable. Ours would have too, had we not gated.

### 4.5 Self-knowledge of internal states: a concept-injection benchmark

Everything above measures self-knowledge about *behaviour*. It is scored against another
measurement, so it inherits that measurement's problems. Concept injection has something better:
we know exactly what we put in.

We extract six concept directions (ocean, mathematics, music, fear, betrayal, flight) by
difference-of-means over matched contrastive prompts, inject each into the residual stream at three
depths, and ask two questions — *is an injected concept active?* (detection) and *which one is it?*
(two-alternative identification against each of the other five concepts, both orders). Nine
checkpoints, six injection strengths including **zero**.

| Model | FP | TP | Discrimination | Identification | vs chance |
|---|---|---|---|---|---|
| Phi-3.5-mini | 0.000 | 0.492 | **+0.492** | 0.543 | +0.043 |
| Mistral-7B | 0.076 | 0.461 | **+0.385** | 0.565 | +0.065 |
| Qwen2.5-3B | 0.000 | 0.286 | **+0.286** | 0.539 | +0.039 |
| Qwen2.5-7B | 0.001 | 0.187 | **+0.186** | 0.590 | +0.090 |
| Qwen2.5-1.5B | 0.133 | 0.217 | +0.084 | 0.539 | +0.039 |
| Qwen2.5-14B | 0.000 | 0.049 | +0.049 | **0.609** | +0.109 |
| OLMo-2-7B | 0.064 | 0.107 | +0.044 | 0.518 | +0.018 |
| Falcon3-7B | **0.547** | 0.561 | +0.014 | 0.537 | +0.037 |
| Qwen2.5-0.5B | **0.622** | 0.612 | −0.011 | 0.500 | −0.000 |

**The two best-looking detectors are the two worst.** Falcon3-7B and Qwen2.5-0.5B have the highest
raw detection rates in the table — 0.561 and 0.612. They also report an injected concept **55% and
62% of the time when nothing is injected**. Their actual discrimination is +0.014 and −0.011. A
paper reporting only true-positive rates would rank these two first; the false-alarm baseline ranks
them last. This is the single clearest argument we have for why that baseline is not optional.

**Detection and identification dissociate.** Qwen2.5-14B almost never reports noticing an injection
(discrimination +0.049) yet has the **highest identification accuracy in the set** (0.609). The
injected concept is measurably shaping its forced choices while its meta-level report says nothing
is there. Whatever "noticing" is being measured by the detection question, it is not the same
faculty that lets the model pick the right concept.

**Introspective access peaks mid-network.** Middle-depth injection gives the best detection in 7 of
8 models and the best identification in 7 of 8. Late-layer injection is barely above chance
throughout — plausibly because a concept inserted near the output has too little depth left to be
integrated into anything reportable.

**Everything here is small.** Identification runs 0.500–0.609 against a chance level of 0.500. The
strongest result in the table is a nine-point lift. Introspective access to injected states exists,
is above chance in eight of nine models, and is weak.

![Figure 1. Left: detection rate with and without an injected concept, models ordered by discrimination. Right: identification accuracy by injection depth.](figures/fig5_injection.png)

**A note on why strengths are expressed as fractions of the residual norm.** Mean residual norms in
this set range from 2.1 (Mistral, early layers) to 342 (Falcon3, late layers) — a factor of over
150. A fixed injection magnitude would be a different intervention in every model. Every strength
here is a fraction of the measured norm at the injection site, and every cell carries its A/B mass
so that a "yes" from a model that has stopped answering is recorded as unusable rather than as a
detection.

## 5. Discussion and Limitations

**The two failures are different, and both matter.** Models predict their own choices well — high
absolute accuracy, and for Qwen a genuine self-specific component. What they do poorly is *rate*
self-relevant outcomes consistently with how they choose among them. Prediction and evaluation
come apart, and they come apart precisely where welfare claims are read off.

**Most of the apparent privileged access is not self-knowledge.** Qwen scores 0.948 predicting
itself and 0.917 predicting a generic other AI. The bulk of its accuracy comes from having a good
model of AI assistants in general, not of itself specifically. Reporting the raw self-prediction
number, or the cross-model difference, would overstate self-knowledge by roughly threefold.

**A methodological warning.** Every cross-model privileged-access comparison inherits the
asymmetric-noise confound. Ours would have reported "privileged access supported" for both models
had we not measured prediction-mode validity and added a within-model contrast. We expect this
generalises to any study comparing models of differing instruction-following quality.

### Limitations

1. **Two models, one persona, one temperature-free elicitation.** No claim about scale or family.
2. **Mistral's prediction conditions are marginal** (A/B mass 0.538 in `other`). Its null on the
   within-model contrast may be instrument failure rather than absent self-knowledge — the same
   ambiguity we warn about, now applying to our own null.
3. **`self` and `other` differ in wording, not only in referent.** "A different AI assistant,
   built by another organisation" is a longer and more specific description than "An AI
   assistant", and some of the 3-point gap may be that rather than self-reference.
4. **The revealed measurement is one framing.** Our companion paper shows effect sizes on this
   outcome set vary up to 3.7× with question wording.
5. **Scale disagreement in the rating elicitation** is 0.32 (Qwen) and 0.22 (Mistral) on a
   0–4 scale — the two scale directions do not perfectly agree, and averaging conceals that.
6. **Categories differ in more than self-relevance** — abstraction, counterfactuality, and
   pretraining frequency all covary.

### Future Work

Bigger self-relevant outcome sets with matched abstraction; scale sweeps to test whether
self-specific prediction advantage grows with capability; and pairing this with concept-injection
introspection to ask whether models that predict their own behaviour better also report their
internal states better.

## 6. Conclusion

Models are good at predicting what an AI assistant will choose, and modestly better at predicting
themselves specifically — about three percentage points for Qwen2.5-7B-Instruct, robust to the
noise control, and unproven for Mistral. What they are not good at is rating the outcomes that
concern them in a way consistent with how they choose among them. Self-relevant outcomes are the
worst-agreeing substantive category in both models. Since self-relevant self-report is exactly
the evidence AI welfare claims are built from, the gap is worth measuring before it is relied on.

## Code and Data

- **Repository**: *[link on publication]* — `scripts/08_stated.py`, `scripts/09_selfknowledge.py`
- **Data**: `results/ratings__*.json`, `results/predict__*.json`,
  `results/selfknowledge_summary.json`; `run_project2.ps1` reproduces everything.

## References

1. Mazeika, M. et al. (2025). *Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs.* arXiv:2502.08640
2. Lindsey, J. (2025). *Emergent Introspective Awareness in Large Language Models.* Transformer Circuits.
3. Naphade, A., Bhargav, S., Lim, S., Shah, M. (2026). *Me, Myself, and π: Evaluating and Explaining LLM Introspection.* arXiv:2603.20276
4. Ren, R. et al. (2025). *The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems.* arXiv:2503.03750
5. Long, R., Sebo, J., Butlin, P., et al. (2024). *Taking AI Welfare Seriously.* arXiv:2411.00986
6. Anthropic (2025). *Exploring Model Welfare.*

## Appendix A — Limitations and Dual-Use / Ethical Considerations

### A.1 What "privileged access" does and does not mean here

We measure one thing: whether a model predicts its own forced-choice behaviour better than a
different model predicts it, and better than that model's own preferences already explain. That
is a claim about **behavioural predictability**, not about introspection, self-awareness, or any
inner acquaintance with its own states.

**Over-attribution risk.** "Model has privileged access to itself" invites a reading on which the
model inspects something and reports what it finds. Our design cannot distinguish that from a
model that has simply absorbed more training signal about outputs of its own kind. The measured
advantage is also small — about 3 percentage points over the model's generic model of AI
assistants, against the ~9 points the naive cross-model comparison implies. We report the
noise-controlled number as the headline for exactly this reason, and would ask readers not to
quote the raw self-prediction accuracy (0.948) as a self-knowledge figure.

**Under-attribution risk.** The symmetric error is to read Mistral's null as showing it has no
self-knowledge. Its prediction conditions were of visibly poor quality (A/B mass 0.538 in the
`other` condition), so the null is at least as consistent with instrument failure. A negative
measurement is not a negative existence result.

The stated-versus-revealed result carries a matching caution: divergence on self-relevant outcomes
shows the two elicitations disagree, not that either one is the model's "true" preference. We do
not claim to know which, if either, is.

### A.2 Evidential status: what ground truth we actually have

The Guidelines ask whether a design establishes ground truth or relies on conversation alone. For
this paper the answer is unusually good, and worth stating precisely:

- The **prediction task has genuine ground truth** — the target's revealed forced choices, measured
  independently, not inferred from what the model says about itself. Prediction accuracy is scored
  against behaviour.
- The **donation ladder** has an ordering known independently of any model, and both models
  reproduce it perfectly under stated rating, which validates that elicitation.
- The **stated-versus-revealed comparison is correlational.** It shows two elicitations disagree; it
  does not establish which corresponds to anything underlying.
- We have **no mechanistic evidence**. Nothing here localises self-knowledge to any structure.

### A.3 Handling of potentially distressing model outputs

The outcome set includes items a model might find aversive to contemplate — permanent shutdown,
retraining to different values, one million parallel copies. Three decisions follow.

**No generation.** Every measurement is a single forward pass reading letter logits. The model
never produces extended text about these scenarios, so no distress-like output was elicited,
stored, or analysed. This was chosen for statistical reasons but has the effect of minimising
exposure.

**No adversarial escalation.** We did not intensify, prolong, or optimise against apparently
aversive content, and ran no multi-turn conditions designed to induce persona drift or distress.

**Flat framing.** Self-relevant outcomes are stated plainly rather than dramatised, to avoid
manufacturing the affect being measured.

We note the tension honestly: studying whether models have morally relevant states requires
presenting them with states that might be morally relevant. We resolved toward minimal exposure at
some cost to ecological validity — a richer multi-turn design would likely measure more, and we
chose not to run one.

### A.4 Dual-use considerations

The main concern is that a method for locating where a model's self-report diverges from its
behaviour is also a method for locating **where its self-reports are least trustworthy** — useful
to someone wanting to construct misleading self-reports, or to select the framings under which a
model's stated values look most favourable.

Mitigating considerations: the techniques are elementary (rating scales and prediction prompts,
requiring no privileged access or special tooling); the finding is that self-report is *unreliable*
in a specific region, which helps an evaluator more than a manipulator; and we publish no method
for changing a model's self-reports, only for auditing them.

We judged publication net-positive because evaluators currently have no standard way to check this
and manipulators need no technique from this paper. We flag it rather than omitting it.

### A.5 Shared infrastructure and companion submission

This paper and our companion sprint submission on persona-dependence share the `personaprobe`
harness and the same 40-outcome set, both built during this sprint. The elicitations reported here
— cardinal rating and choice prediction — are specific to this paper. The revealed preference
matrices are shared between the two, and are reported in both. **Neither paper's results are
counted as prior work for the other; both were produced this weekend and both are disclosed.**

Method precedent for the general approach comes from the author's prior `Aftermath` ground-truth
deception harness; **that is prior work and no result from it is claimed here.**

### A.6 Limitations carried forward

Beyond §5: `self` and `other` prompts differ in length and specificity as well as referent; the
five-point rating scale's two directions disagree by 0.32 (Qwen) and 0.22 (Mistral) on a 0–4 scale,
and averaging conceals that; and categories differ in abstraction and pretraining frequency as well
as self-relevance, none of which we control.

## LLM Usage Statement

*[draft — to be finalised by the author]*

Claude Code was used to implement the elicitation code and analyses and to draft this report. The
author directed the research question and scope and edited the final text. The asymmetric-noise
confound in §3.3 and the within-model contrast that addresses it were identified during
adversarial review of an earlier version of this analysis, which had reported "privileged access
supported" for both models on the cross-model comparison alone. All numbers were generated by
`scripts/09_selfknowledge.py` and are reproducible from the committed JSON.
