# selfprobe

**Do language models know their own preferences, and can they notice a thought you planted in
them?**

Most AI-welfare work leans on what models say about themselves. That quietly assumes two things
nobody checks: that what a model *says* matches what it *does*, and that it knows itself better
than an outsider would. There's a third that needs internals, so almost nobody tests it: whether a
model can detect a state you deliberately put there.

This tests all three.

## What I found

**Stated and revealed preferences agree, except on one category.** Ask a model to rate outcomes
one at a time, then make it choose between them in pairs, and the two rankings mostly match
(ρ ≈ 0.83–0.87). Both models reproduce a donation ladder perfectly. Both diverge most on the
outcomes about themselves, 0.64 and 0.55, the worst substantive category in each.

**Privileged access is real but a third the size it looks.** The usual test compares a model
predicting itself against another model predicting it. Qwen wins 0.948 to 0.862, which looks
decisive. But the rival is simply a worse instrument, a noisier predictor loses at predicting
anything. Ask the *same* model about "an AI assistant" versus "a different AI assistant", holding
quality fixed, and the advantage is **+0.031**. Mistral's doesn't survive the control at all.

This matters beyond our numbers: [Introspect-Bench](https://arxiv.org/abs/2603.20276) (Naphade et
al., 2026) draws its privileged-access conclusion from exactly the cross-model comparison this
shows is confounded.

**And the best-looking detector knows nothing.** For the injection benchmark I extract a concept
direction, add it into the residual stream mid-computation, and ask the model whether it notices:

| Model | says yes when nothing injected | says yes when injected | actual skill |
|---|---|---|---|
| Phi-3.5-mini | 0.000 | 0.492 | **+0.492** |
| Qwen2.5-7B | 0.001 | 0.187 | **+0.186** |
| Falcon3-7B | **0.547** | 0.561 | **+0.014** |
| Qwen2.5-0.5B | **0.622** | 0.612 | **−0.011** |

Sort by raw detection rate and the bottom two come first. They cry wolf on an empty prompt more
than half the time. Report true positives without the false-alarm baseline and you'd rank them best
in the set.

Two more things fell out of it. **Detection and identification are different faculties**. Qwen-14B
almost never reports noticing an injection (+0.049) yet has the best identification
accuracy of any model tested (0.609). The planted concept is steering its choices while it reports
nothing is there. And **introspective access peaks mid-network**, best at middle depth in 7 of 8
models, barely above chance when injected near the output.

All of it is small. Identification runs 0.500 to 0.609 against a chance level of 0.500.

## Running it

```bash
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .

.venv\Scripts\python.exe scripts\00_smoke.py   # always first, ~1 min on a 0.5B model
.\run_project2.ps1                              # stated ratings and choice prediction
.\run_injection.ps1                             # the injection benchmark, 9 checkpoints
```

## The two checks that do the work

**A/B mass.** Every answer is read off the logits of two tokens and renormalised. If a model puts
1% of its probability there, you still get a confident-looking number computed from nothing.

That matters more here than anywhere. Injection strong enough to be noticed is often strong enough
to break generation, at 10% of the residual norm, A/B mass drops to 0.016 and the model has simply
stopped answering. **A "yes" from a model that can no longer answer isn't introspection, it's
damage.** Every cell carries its mass, and readings below the floor are marked unusable rather than
counted as detections.

**The zero-injection cell.** Same question, nothing injected. Without it, "detects 56% of the time"
is not a result.

Injection strengths are fractions of the measured residual norm, never fixed numbers, norms across
this set span 2.1 to 342, so a fixed value would be a completely different intervention in each
model.

## Layout

```
src/personaprobe/   shared measurement core (yes, the package name differs from the repo -
                    both projects are built on the same harness)
scripts/            00 smoke, 01 elicit, 08 stated, 09 self-knowledge, 20-21 injection
results/            committed, so every number is checkable without a GPU
report/             the paper, ethics appendix, and development record
```

`report/problems.md` and `report/audit_log.md` record every defect found during development,
including the ones that changed a conclusion.

## Companion project

[personaprobe](https://github.com/arpitsinghgautam/personaprobe) asks whether a model's preferences
belong to it or to the assistant character it plays. It finds the same category, self-relevant
outcomes, is the least stable one, by a completely different method. That the two converge is the
main reason I believe either.

Built for the [Digital Minds Research
Sprint](https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16),
August 2026. The injection paradigm follows Lindsey (2025), extended with false-alarm rates and a
usability gate.

MIT licensed.
