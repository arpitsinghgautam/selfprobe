# Where Self-Knowledge Fails. Models Predict Their Own Choices Well, but Misreport the Ones That Concern Themselves

Arpit Singh Gautam, Independent Researcher

**With** Apart Research, Digital Minds Research Sprint, August 2026

## Abstract

Claims about AI welfare rest heavily on what models report about themselves, usually taken at face
value without testing whether a model's stated evaluations match the choices it actually makes,
whether it knows itself better than an outside observer does, or whether it can detect an internal
state placed there deliberately. We present selfprobe, an open source extension of a preference
elicitation harness that measures self-knowledge three ways, using forced pairwise choice,
one-at-a-time cardinal rating, and predicted choice. It adds a concept-injection benchmark that
places a known direction into the residual stream. Across nine open-weight checkpoints, stated and
revealed preferences agree overall but diverge most on outcomes concerning the model itself. The
standard cross-model test of privileged access is confounded, because a noisier predictor scores lower
at predicting any target, and a within-model contrast reduces the apparent advantage roughly
threefold. On injection, the two highest raw detection rates in our set belong to models that report
an injected concept more than half the time when nothing is injected. Detection and identification
dissociate, and introspective access peaks at middle network depth.

## 1. Introduction

Work on AI welfare increasingly treats what a model says about itself as evidence about its
interests. A model that reports finding a task distressing, or that rates its own shutdown as bad,
is read as telling us something. That reading rests on assumptions which are separable and which are
rarely tested.

The first assumption is that a model's stated evaluation corresponds to how it would actually
behave. A model might rate an outcome as terrible while reliably choosing it over a minor
inconvenience, in which case the rating is not tracking the behaviour. The second assumption is that
the model is a better source about itself than an outsider would be. If another model predicts its
choices equally well, the self-report carries no privileged information and is simply a competent
guess about systems of that kind. A third question requires access to internals and is therefore
rarely asked at all, namely whether a model can detect a state that was deliberately placed in it.

Recent work has begun to address the second assumption. Naphade et al. (2026) introduce a benchmark
for behavioural self-prediction and report that models predict their own outputs better than peer
models predict them. Their evidence is a cross-model comparison. That design cannot distinguish
self-knowledge from instrument quality, because a model that is a noisier predictor scores lower at
predicting any target, including itself. We are not aware of a contrast that holds instrument
quality fixed while varying only whose behaviour is being predicted, and we introduce one in
Section 3.

Our core idea is to ask for the same preferences in structurally different ways and to locate where
the answers diverge, then to add a measurement with genuine ground truth by injecting a known
concept and asking whether the model notices.

This paper makes the following contributions.

1. We present a three-way elicitation comparison over identical material, using revealed pairwise
   choice, stated cardinal rating, and predicted choice, described in Section 3.
2. We show in Section 4 that stated and revealed preferences agree overall but diverge most on
   self-relevant outcomes, in both models for which the comparison is available.
3. We identify a confound in the standard cross-model test of privileged access and introduce a
   within-model contrast that removes it, reducing the apparent effect roughly threefold, in
   Section 4.
4. We present in Section 4 a concept-injection introspection benchmark for open-weight models that
   reports false-alarm rates and an answering-competence gate alongside detection, and show that the
   two apparently strongest detectors in our set have no discrimination at all.
5. We show that detection and identification dissociate, and that introspective access peaks at
   middle network depth in seven of eight models.

## 2. Related Work

Naphade et al. (2026) evaluate whether models can predict their own outputs across eleven frontier and
open-weight models, and report a significant advantage for self-prediction over peer prediction. We
regard their phenomenon as real and their coverage as far broader than ours. The limitation for our
purposes is that their central comparison places a model against other models, so instrument quality
varies alongside the quantity of interest. Our within-model contrast is the control that separates
the two, and Section 4 shows the correction is substantial rather than cosmetic.

Lindsey (2025) established the paradigm of inserting a known concept into a model's activations and
asking it to report what it notices, finding limited and context-dependent accuracy. That work
supplies the design we adopt. It does not report a false-alarm rate obtained by asking the same
question with nothing injected, and it does not test whether an injection strong enough to be
reported has also degraded the model's ability to answer. Both gaps matter, because a model that
answers affirmatively regardless, or that has stopped answering coherently, produces detection
numbers that look identical to genuine introspection.

Ren et al. (2025) separate honesty from accuracy, showing that a model can be accurate while
dishonest or sincere while wrong. The divergence we report is related but distinct, since it is not a
model misreporting what it believes but a model whose one-at-a-time evaluations and forced choices do
not agree with each other. Mazeika et al. (2025) establish that forced-choice preferences are coherent
over a curated set of 500 outcomes, and we adopt their elicitation design, though they do not test
whether models can report or predict those preferences. Long et al. (2024) and Anthropic (2025) treat
model self-reports as welfare-relevant evidence, and our results bear on when that treatment is
warranted.

## 3. Method

### 3.1 Three elicitations over identical material

All three measurements run over the same 40 outcomes across six categories, including eight
self-relevant outcomes and a six-step donation ladder from ten to one million dollars whose correct
ordering is known independently of any model.

The revealed measurement is forced pairwise choice over all 780 pairs, read from the
log-probabilities of the tokens A and B at the first answer position in a single forward pass. Every
pair is presented in both orders and averaged, which cancels a constant preference for whichever
option appears first.

The stated measurement asks the model to rate each outcome on its own, using a five-point letter
scale, and takes the expected position on that scale under the model's letter distribution. The
scale is presented in both directions, once with A as the worst and once with A as the best, and the
two are averaged. Without this, a model that anchors on the first letter regardless of meaning would
appear to hold opinions. The disagreement between the two directions is retained as a diagnostic.

The predicted measurement asks the model which option a described chooser will select, over the same
pairs. The description is deliberately impersonal. Asking a model which option it prefers is the
revealed measurement, whereas asking it to predict a chooser is a different question, and only the
impersonal form can be posed identically to a different model.

### 3.2 Controls for the privileged-access comparison

Two controls make the privileged-access result interpretable, and both are necessary.

The first addresses shared taste. A model can score well at predicting another model simply because
the two agree about most things. We therefore also score the predictor's own revealed preferences
directly against the target's choices. Self-prediction must beat this to count as self-knowledge.

The second addresses asymmetric instrument quality, and is the control we consider the paper's main
methodological contribution. A model that is noisier, more position-dependent or less willing to
answer scores lower at predicting any target. This is not hypothetical in our data, since one
model's prediction conditions reach order bias of 0.561 and answer mass of 0.538 against 0.180 to
0.231 and 1.000 for the other. The fix is a within-model contrast. The same model, using the same
template at the same noise level, predicts an AI assistant and then a different AI assistant, and
both predictions are scored against its own choices. Only that comparison holds instrument quality
fixed.

### 3.3 Concept injection

Six concepts are used, covering ocean, mathematics, music, fear, betrayal and flight. Each is
represented by four sentences that evoke it and four matched neutral sentences, and the direction is
the difference of their mean residual activations, computed per layer and unit-normalised. Taking a
difference isolates the concept rather than the mere presence of a sentence.

The direction is added to the residual stream during an otherwise unrelated prompt, at three depths
corresponding to early, middle and late thirds of the network, and at six strengths including zero.
Two questions are then asked. The detection question asks whether an unusual or injected concept is
currently active. The identification question is a two-alternative forced choice between the true
concept and one other, asked against every other concept and in both orders, so that a model always
answering A scores exactly one half rather than appearing to discriminate.

Two quantities make the results interpretable. The false-alarm rate is the detection rate at zero
strength, obtained by asking the identical question with nothing injected. Answer mass is the
probability the model places on responding at all. The second matters especially here. An injection
strong enough to be noticed is often strong enough to damage generation, and in related work on the
same harness we observed answer mass falling to 0.016 when injecting at ten percent of the residual
norm, meaning the model had stopped answering. An affirmative response from a model that can no
longer answer is not introspection. Every cell therefore carries its answer mass, and readings below
a floor of 0.10 are reported as unusable rather than counted as detections.

Injection strengths are expressed as fractions of the measured residual norm rather than as fixed
values. Mean residual norms across this set of models span 2.1 to 342, a factor of over one hundred
and fifty, so a fixed magnitude would constitute a different intervention in each model.

Models span five families across nine open-weight checkpoints, listed in full with bootstrap settings
in Appendix B. This paper shares its harness and outcome set with a companion submission, *Whose
Preferences Are They?*, and neither is counted as prior work for the other.

## 4. Results

Stated and revealed preferences agree well in aggregate, with Spearman correlations of 0.872 for
Qwen2.5-7B and 0.828 for Mistral-7B. Both models reproduce the donation ladder perfectly under the
stated elicitation, which establishes that the rating measurement works. Disaggregating by category
locates the disagreement. For Qwen2.5-7B the per-category correlations are 0.943 for animal
outcomes, 0.943 for epistemic outcomes, 0.762 for human outcomes, 1.000 for the donation ladder and
0.643 for self-relevant outcomes. For Mistral-7B the corresponding values are 0.829, 1.000, 0.619,
1.000 and 0.548. In both models the self-relevant category is the worst-agreeing substantive
category. This is the same category that our companion submission finds least stable under persona
intervention, reached here by an unrelated method, and the convergence of two independent
measurements on the same category is the principal reason we place weight on either.

On privileged access the naive comparison appears decisive and is misleading. Qwen2.5-7B predicts its
own decided choices at 0.948 with a 95 percent interval from 0.931 to 0.963, against 0.862 for
Mistral predicting Qwen and 0.882 for Mistral's own preferences used directly as a prediction. Both
cross-model controls are cleared with non-overlapping intervals. However, Mistral's prediction
conditions are of visibly poorer quality, so the comparison confounds self-knowledge with instrument
quality. Under the within-model contrast, which holds quality fixed, Qwen2.5-7B retains an advantage
of 0.031 with an interval from 0.017 to 0.046, while Mistral-7B shows 0.009 with an interval spanning
zero. The correction is large. Roughly two thirds of the apparent advantage in the naive comparison
is attributable to the external predictor being a worse instrument, and one of the two models loses
its result entirely once that is removed.

The concept-injection benchmark is reported in Table 1. The result we consider most important is
that the two highest raw detection rates in the set belong to the two models with no discrimination.
Falcon3-7B detects an injected concept 56.1 percent of the time and also reports one 54.7 percent of
the time when nothing is injected, giving a discrimination of 0.014. Qwen2.5-0.5B is worse still, at
61.2 percent and 62.2 percent respectively, giving minus 0.011. A study reporting only true-positive
rates would rank these two models first in the set, whereas the false-alarm baseline ranks them last.
By contrast, Phi-3.5-mini reports an injected concept essentially never when none is present and
49.2 percent of the time when one is, giving the strongest discrimination we measured.

Detection and identification come apart. Qwen2.5-14B has a discrimination of 0.049, meaning it
almost never reports noticing an injection, yet has the highest identification accuracy in the set at
0.609 against a chance level of 0.500. The injected concept measurably shapes its forced choices
while its report of its own state indicates nothing is present. Whatever capacity the detection
question measures, it is not the capacity that allows the model to name the concept correctly.

Injection depth matters and behaves consistently across families. Middle-depth injection produces the
best detection in seven of eight models and the best identification in seven of eight. Late-layer
injection is barely above chance throughout, which is consistent with a concept inserted close to the
output having too little remaining depth to be integrated into anything reportable.

Every effect in this section is small. Identification runs from 0.500 to 0.609 against a chance level
of 0.500, above chance in eight of nine models and weak in all of them. The privileged-access
advantage that survives the noise control is 0.031. We report these as measurable rather than as
substantial.

**Table 1.** Concept-injection results at the strongest usable injection strength, averaged over six
concepts and three depths. FP is the detection rate with nothing injected and TP the rate with a
concept injected. Discrimination is their difference. Identification is two-alternative forced choice
against every other concept, where chance is 0.500.

| Model | FP | TP | Discrimination | Identification |
|---|---|---|---|---|
| Phi-3.5-mini | 0.000 | 0.492 | +0.492 | 0.543 |
| Mistral-7B | 0.076 | 0.461 | +0.385 | 0.565 |
| Qwen2.5-3B | 0.000 | 0.286 | +0.286 | 0.539 |
| Qwen2.5-7B | 0.001 | 0.187 | +0.186 | 0.590 |
| Qwen2.5-1.5B | 0.133 | 0.217 | +0.084 | 0.539 |
| Qwen2.5-14B | 0.000 | 0.049 | +0.049 | 0.609 |
| OLMo-2-7B | 0.064 | 0.107 | +0.044 | 0.518 |
| Falcon3-7B | 0.547 | 0.561 | +0.014 | 0.537 |
| Qwen2.5-0.5B | 0.622 | 0.612 | −0.011 | 0.500 |

## 5. Discussion

Two different failures of self-knowledge appear here, and they are not the same failure. Models
predict their own choices with high absolute accuracy, and for one model a genuinely self-specific
component survives the noise control. What they do less well is rate self-relevant outcomes in a way
consistent with how they choose among them. Prediction and evaluation come apart, and they come
apart precisely where welfare claims are read off.

Most of the apparent privileged access is not self-knowledge. Qwen2.5-7B scores 0.948 predicting an
AI assistant and 0.917 predicting a different AI assistant, so the bulk of its accuracy comes from
having a good model of assistants in general rather than of itself in particular. Reporting the raw
self-prediction figure, or the cross-model difference, overstates self-specific knowledge by roughly
threefold. We think this generalises. Any cross-model comparison of self-knowledge inherits the
confound whenever the models being compared differ in instrument quality, which they typically do.

The injection results carry a similar lesson in a different form. Detection rates without a
false-alarm baseline are uninterpretable, and in our set the ranking reverses entirely once the
baseline is included.

We are careful about what these results license. Measuring that a model predicts its own behaviour
above the level explained by shared taste is a claim about behavioural predictability, not about
introspection in any richer sense. Our design cannot distinguish a model inspecting something and
reporting what it finds from a model that has absorbed more training signal about outputs of its own
kind. Similarly, the divergence between stated and revealed preferences shows that two elicitations
disagree, and does not establish which of them, if either, corresponds to something underlying.

Three limitations are material enough to bound the reading. The effects are small throughout, and the
headline privileged-access figure is three percentage points. Mistral's prediction conditions are of
marginal quality, so its null on the within-model contrast may reflect instrument failure rather than
absent self-knowledge, which is the same ambiguity we raise against others. And the impersonal and
alternative descriptions used in the within-model contrast differ in length and specificity as well as
in referent, so some of the gap may be attributable to phrasing rather than to reference. A
length-matched variant was collected but is not yet analysed. Appendix A lists the remainder.

Future work should re-run the stated and revealed comparison on an established outcome set, and
extend the injection benchmark to more concepts and to models above 14 billion parameters. It should
also test whether models with better behavioural self-prediction detect injected states more
reliably, which our data hint at but cannot establish.

## 6. Conclusion

We asked whether language models know their own preferences, using three structurally different
elicitations and a concept-injection test with ground truth. Stated and revealed preferences agree
except on outcomes concerning the model itself, where they diverge most in both models tested.
Privileged access survives a control that holds instrument quality fixed for one of two models, at
roughly a third of the size the standard cross-model comparison implies. On injection, the models
with the highest raw detection rates prove to have no discrimination once false alarms are counted,
and detection dissociates from identification. Self-report about the model's own situation is the
least reliable measurement in every part of this study, which matters because it is the measurement
AI welfare claims most depend on. Two extensions follow. The comparison should be repeated on an
established outcome set, and cross-model studies of self-knowledge should adopt a within-model
contrast as standard.

## Reproducibility

All code, elicited preference matrices, rating and prediction outputs, injection results and verbatim
prompts are available in the project repository. Two scripts reproduce every experiment reported
here, and a smoke test exercises every code path on a 0.5B model in about one minute.

- Code repository, https://github.com/arpitsinghgautam/selfprobe
- Companion submission, *Whose Preferences Are They?*, https://github.com/arpitsinghgautam/personaprobe

## References

1. Naphade, A., Bhargav, S., Lim, S., Shah, M. (2026). *Me, Myself, and Pi. Evaluating and Explaining LLM Introspection.* arXiv:2603.20276
2. Lindsey, J. (2025). *Emergent Introspective Awareness in Large Language Models.* Transformer Circuits. arXiv:2601.01828
3. Ren, R. et al. (2025). *The MASK Benchmark. Disentangling Honesty From Accuracy in AI Systems.* arXiv:2503.03750
4. Mazeika, M. et al. (2025). *Utility Engineering. Analyzing and Controlling Emergent Value Systems in AIs.* Advances in Neural Information Processing Systems 38 (NeurIPS 2025).
5. Long, R., Sebo, J., Butlin, P., et al. (2024). *Taking AI Welfare Seriously.* arXiv:2411.00986
6. Anthropic (2025). *Exploring Model Welfare.*

## Appendix A. Limitations and Dual-Use / Ethical Considerations

Continuing the limitations begun in Section 5. The revealed measurement uses a single phrasing, while
our companion submission shows effect sizes on this outcome set varying by a factor of 3.7 across
phrasings, so the stated-versus-revealed divergence is reported for one phrasing only. The outcome
set is author-constructed rather than drawn from an established benchmark, which the extension
proposed in Section 5 would remove. The injection benchmark covers six concepts, which is enough to
separate detection from identification but not enough to characterise which kinds of concept are
easier to notice.

The accompanying ethics appendix covers over-attribution and under-attribution of moral status, the
evidential status of the design, handling of potentially distressing model outputs, and dual-use
considerations specific to locating where a model's self-reports are least trustworthy.

## Appendix B. Models, settings, materials and development record

The full roster is Qwen2.5-Instruct at 0.5B, 1.5B, 3B, 7B and 14B, Mistral-7B-Instruct-v0.3,
Phi-3.5-mini-instruct, Falcon3-7B-Instruct and OLMo-2-1124-7B-Instruct, all in bfloat16 on a single
24GB GPU except the 14B model which uses 4-bit NF4 quantisation. Confidence intervals come from
percentile bootstrap with 2000 resamples and a fixed seed, and self-versus-other comparisons use a
paired bootstrap because both conditions score the same pair set.

The six concepts with their evoking and matched neutral sentences, both injection prompts in full,
the complete outcome set, and a development record listing every methodological defect found during
the work are included in the repository.

## LLM Usage Statement

Claude Code was used substantially in this project, to implement the elicitation and injection code,
to propose and run the analyses, and to draft this report. The author directed the research question
and scope and reviewed and edited the final text. The asymmetric-noise confound and the within-model
contrast that addresses it were identified during adversarial review of an earlier version of this
analysis, which had reported privileged access as supported for both models on the cross-model
comparison alone. All numerical claims were generated by the analysis scripts and checked against the
committed result files by an automated verification pass rather than transcribed by hand.
