# Submission form content, ready to paste

Two submissions with unique titles, which is required when submitting more than one project.
Everything below is final copy. Titles match the papers exactly, so resubmitting under the same
title replaces the files rather than creating a third entry.

---

## SUBMISSION 1

### Project Title*
```
Whose Preferences Are They? Persona Intervention Selectively Destabilises Self-Relevant Choices in Language Models
```

### Project Summary*
```
Language models express coherent, transitive preferences, and AI-welfare research increasingly reads
them as evidence about model interests. Text alone cannot distinguish the model's preferences from
the assistant character's. We built personaprobe, an open-source harness that re-runs any preference
measurement under persona intervention, covering identity swaps, affect suppression and mechanistic
ablation, and reports how much survives. On Qwen2.5-7B-Instruct aggregate preferences look nearly
persona-invariant at 0.029, but that invariance is carried entirely by outcomes the model has no
stake in. Preferences over its own shutdown, retraining and memory are 0.21 to 0.29 less stable than
every other category, surviving controls for utility spacing and measurement noise. Stripping the
model's affect leaves them intact at 0.924; replacing its identity collapses them to 0.436.
Rewriting the same outcomes in the third person more than doubles the effect, ruling out a pronoun
artifact. Only twelve of twenty-two model and phrasing combinations pass our validity criteria, and
the effect is absent in two families that pass them.
```

### Upload your PDF report*
`report/report_4page.pdf`

### Are you interested in publishing this project?*
**Yes.** Required for the funder-review path.

### Tracks*
- **Track 5**, The Assistant Persona and Model Identity *(primary)*
- **Track 1**, Model Preferences and Trade-offs
- **Track 4**, Preference Elicitation Methods

### Optional uploads
| Field | Value |
|---|---|
| Presentation Recording | https://youtu.be/9ZozlD5c7L0 (unlisted, 5:08) |
| Project Code | https://github.com/arpitsinghgautam/personaprobe |
| Slideshow | `report/deck_p1_simple.pdf` |
| Project image | `figures/project_image.png` |
| Additional Material | https://github.com/arpitsinghgautam/personaprobe/blob/main/report/audit_log.md |

---

## SUBMISSION 2

### Project Title*
```
Where Self-Knowledge Fails. Models Predict Their Own Choices Well, but Misreport the Ones That Concern Themselves
```

### Project Summary*
```
AI-welfare claims rest on two untested and separable assumptions, that a model's stated evaluations
match the choices it actually makes, and that it knows itself better than an outside observer does.
We test both using three independent elicitations over identical material, namely forced pairwise
choice, one-at-a-time cardinal rating, and predicted choice, and add a concept-injection benchmark
with ground truth. Stated and revealed preferences agree well overall, at 0.872 and 0.828, but
diverge sharply on outcomes concerning the model itself, the lowest-agreeing substantive category in
both models at 0.643 and 0.548. The standard cross-model test of privileged access is confounded,
because a noisier external predictor scores lower at predicting any target. Under a within-model
contrast holding instrument quality fixed, Qwen2.5-7B retains a small advantage of 0.031 and
Mistral-7B does not. On injection, the two highest raw detection rates belong to models that report
an injected concept more than half the time when nothing is injected.
```

### Upload your PDF report*
`report/report2.pdf`

### Are you interested in publishing this project?*
**Yes.**

### Tracks*
- **Track 3**, Introspection and Self-Report Reliability *(primary)*
- **Track 1**, Model Preferences and Trade-offs
- **Track 4**, Preference Elicitation Methods

### Optional uploads
| Field | Value |
|---|---|
| Presentation Recording | https://youtu.be/vLIvg1qGzxU (unlisted, 6:05) |
| Project Code | https://github.com/arpitsinghgautam/selfprobe |
| Slideshow | `report/deck_p2_simple.pdf` |
| Project image | `figures/project_image_2.png` |
| Additional Material | https://github.com/arpitsinghgautam/selfprobe/blob/main/report/audit_log.md |

---

## Team details, identical on both

| Field | Value |
|---|---|
| Team Name* | `Arpit Singh Gautam` |
| Location* | `Bengaluru, India` |
| Team Member Name* | `Arpit Singh Gautam` |
| Affiliation | Independent Researcher |

---

## Pre-submission checklist

Run this **per submission**, immediately before uploading.

- [x] **Both GitHub repos PUBLIC**, verified by anonymous fetch
- [x] Both videos uploaded unlisted, both links verified reachable anonymously
- [ ] PDF built from the official template, opens correctly, under 25 MB
- [ ] Abstract within the 150-word guideline
- [ ] Author name and affiliation present
- [ ] Limitations and Dual-Use / Ethical Considerations appendix present
- [ ] Prior-work disclosure present, with Aftermath and the companion submission both declared
- [ ] Every link in the PDF resolves
- [ ] Titles unique between the two submissions and identical to the papers
- [ ] Publishing opt-in set to **Yes**
- [ ] Tracks selected as above
- [ ] Confirmation email received; if not within a few hours, email sprints@apartresearch.com

**Submit early.** Resubmitting under the exact same title replaces the files at no cost, so an early
safety submission removes deadline risk entirely.

---

## After submitting

A funding opt-in form arrives by email. Ask **$20 to 50K**, against the roadmap in `decisions.md`.

1. Elicitation methods that work on weak instruction-followers and base checkpoints. This is the
   largest measured gap, since Mistral was partly unmeasurable and the base checkpoint failed at its
   own baseline
2. Extension to valence and distress measurements
3. Extension to introspection probes under the same validity gating
4. Distributed and non-linear mechanistic interventions, which the single-direction ablation null
   says are needed
5. Pre-registration and replication infrastructure

The pitch is not that we found something about personas. It is that **four results across these two
papers would have been published wrong without diagnostics this field does not currently run.**
