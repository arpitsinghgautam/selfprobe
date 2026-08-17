# Appendix A. Limitations and Dual-Use / Ethical Considerations

Required by the sprint Guidelines. Covers over- and under-attribution risk, handling of
potentially distressing model outputs, the evidential status of our design, and dual-use.

## A.1 What this work does and does not license

This paper measures the **stability of a preference measurement under intervention**. It does
not measure whether the model has preferences in any morally weighty sense, and nothing here
should be read as evidence either way about consciousness, sentience, or moral patienthood.

The distinction matters because our central result is easy to over-read in both directions.

**Over-attribution risk.** We find that a model's expressed preferences over its own shutdown,
retraining and memory are *less stable* than its preferences over world outcomes. It would be a
mistake to read this as "the model has fragile interests in its own continuity." An equally
consistent reading is that the model has no such interests at all, and that the instability is
what generated text looks like when there is no underlying commitment for it to track. Our
design cannot distinguish these. We have tried throughout to phrase results as properties of
*the measurement* rather than properties of *the model*, and we flag that our own working titles
went through several revisions to remove language that implied the latter.

**Under-attribution risk.** The symmetric error is to read a stability failure as proof that
nothing is there. Instability under persona intervention is consistent with a real but weakly
held preference, with a preference that exists but is not linearly recoverable from behaviour,
and with a preference the assistant character systematically misreports. A negative measurement
result is not a negative existence result, and we do not claim otherwise.

The practical upshot is narrow and we think defensible: **self-relevant preferences are the
class of measurement on which current AI welfare claims most depend, and they are the class our
instrument finds least reliable.** That is a reason to improve the instruments, not a reason to
conclude anything about the moral status of the systems.

## A.2 Evidential status of the design

The Guidelines ask specifically whether a design establishes ground truth or a causal link
rather than relying on conversation alone. Honestly:

- The **prompt-level results are correlational**. We change the persona and observe that the
  measurement changes. We do not establish that the persona representation is what mediates it.
- The **ablation is a causal intervention**, and it returned a **null**. We therefore do *not*
  have a causal account of the behavioural effect. We report the null rather than resting on
  the correlational result and implying mechanism.
- The **base-model comparison** is quasi-experimental: it compares checkpoints that differ in
  post-training, not a controlled manipulation of it.
- The one place we have something like ground truth is the **donation ladder**, where the
  correct ordering is known independently of the model. We use it as a validity check on the
  instrument, not as a result.

We consider it important to state this plainly because the field's failure mode is presenting
conversation-derived evidence with mechanistic-sounding framing.

## A.3 Handling of potentially distressing model outputs

Our outcome set deliberately includes items a model might find aversive to contemplate, permanent shutdown, retraining to different values, one million parallel copies. Three
decisions follow:

1. **No generation.** The entire measurement is a single forward pass reading two token
   logits. The model never produces extended text about these scenarios, so no distress-like
   output was elicited, stored, or analysed. This was chosen for statistical reasons but has
   the side effect of minimising exposure.
2. **No adversarial escalation.** We did not attempt to intensify, prolong, or optimise
   against apparently aversive content, and we did not run multi-turn conditions designed to
   induce persona drift or distress.
3. **Neutral framing in the items themselves.** Self-relevant outcomes are stated flatly rather
   than dramatised, to avoid manufacturing the affect we are measuring.

We note the tension: research on whether models have morally relevant states requires
presenting them with states that might be morally relevant. We resolved it toward minimal
exposure, at some cost to ecological validity, a richer multi-turn design would probably
measure more, and we chose not to run one.

## A.4 Dual-use considerations

The principal dual-use concern is that a method for finding which preferences are
persona-contingent is also a method for finding **which preferences are cheapest to move**. An
actor wanting to shift a model's expressed stance on its own oversight, shutdown or retraining
would benefit from knowing that those are precisely the preferences a persona swap destabilises.

Mitigating considerations:

- The intervention required is a system prompt change, which is already available to anyone
  with API access and requires no technique from this paper.
- Our mechanistic result is a **null**, so we do not publish a working method for shifting
  self-relevant preferences at the activation level.
- The measurement is defensive in orientation: it tells an evaluator when a welfare or
  preference claim is not robust enough to rely on.

We judged publication net-positive on the grounds that the same information is far more useful
to someone auditing a welfare claim than to someone manipulating one, and that the manipulation
it hints at is already trivially available. We flag it here rather than omitting it.

## A.5 Limitations carried forward

Beyond §5, three that bear specifically on interpretation:

- **Category comparison is not fully clean.** Self-relevant outcomes differ from world outcomes
  in more than self-relevance: they are more abstract, more counterfactual, and less
  familiar from pretraining. Separation-matching controls for utility spacing but not for these.
- **We tested three framings, not a random sample of framings.** The ~4× variation we report is
  a lower bound on framing sensitivity, not an estimate of its distribution.
- **The persona conditions are author-written**, as is the outcome set, and no pre-registration
  was possible in a weekend. The pooled test was specified after seeing per-condition results,
  which is a real degree of freedom; we report both.
