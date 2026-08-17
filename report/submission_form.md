# Submission form content — ready to paste

Two submissions, unique titles (required when submitting more than one project).
Everything below is final copy; no placeholder text.

---

## SUBMISSION 1

### Project Title*
```
Whose Preferences Are They? Persona Intervention Selectively Destabilises Self-Relevant Choices in LLMs
```

### Project Summary*
```
Frontier models express coherent, transitive preferences, and AI-welfare research increasingly
reads them as evidence about model interests. But text alone cannot distinguish the model's
preferences from the assistant character's. We built personaprobe, an open-source harness that
re-runs any preference measurement under persona intervention — identity swaps, affect suppression,
and mechanistic ablation — and reports how much survives. On Qwen2.5-7B-Instruct, aggregate
preferences are near persona-invariant (0.03), but that invariance is carried entirely by outcomes
the model has no stake in. Preferences over its own shutdown, retraining and memory are 0.22–0.31
less stable than every other category, surviving controls for utility spacing and measurement
noise. Stripping the model's affect leaves them intact; replacing its identity collapses them.
Effect size varies 3.7× with question wording alone.
```

### Upload your PDF report*
`report/report_4page.pdf`

### Are you interested in publishing this project?*
**Yes** — required for the funder-review path.

### Tracks*
- **Track 5** — The Assistant Persona & Model Identity *(primary)*
- **Track 1** — Model Preferences & Trade-offs
- **Track 4** — Preference Elicitation Methods

### Optional uploads
| Field | Value |
|---|---|
| Presentation Recording | *(video URL — Phase C)* |
| Project Code | *(GitHub URL — must be PUBLIC before submitting)* |
| Slideshow | `report/deck.pdf` |
| Project image | `figures/project_image.png` |
| Additional Material | `<repo>/blob/main/report/audit_log.md` |

---

## SUBMISSION 2

### Project Title*
```
Where Self-Knowledge Fails: Models Predict Their Own Choices Well, but Misreport the Ones That Concern Themselves
```

### Project Summary*
```
AI-welfare claims rest on two untested assumptions: that a model's stated evaluations match the
choices it actually makes, and that it knows itself better than an outside observer does. We test
both on two open-weight models using three independent elicitations — forced pairwise choice,
one-at-a-time cardinal rating, and predicted choice. Stated and revealed preferences agree well
overall (rho 0.83–0.87) but diverge sharply on outcomes concerning the model itself, the
lowest-agreeing substantive category in both models. On privileged access we show the standard
cross-model test is confounded: a noisier external predictor scores lower regardless of
self-knowledge. Under a within-model contrast holding instrument quality fixed,
Qwen2.5-7B-Instruct retains a small but reliable advantage (+0.031, 95% CI [+0.017, +0.046]);
Mistral-7B-Instruct-v0.3 does not.
```

### Upload your PDF report*
`report/report2.pdf`

### Are you interested in publishing this project?*
**Yes**

### Tracks*
- **Track 3** — Introspection & Self-Report Reliability *(primary)*
- **Track 1** — Model Preferences & Trade-offs
- **Track 4** — Preference Elicitation Methods

### Optional uploads
Same video, code, slideshow and additional-material links as Submission 1.
Project image: `figures/project_image_2.png`

---

## Team Details (identical on both)

| Field | Value |
|---|---|
| Team Name* | `Arpit Singh Gautam` |
| Location* | `Bengaluru, India` |
| Team Member Name* | `Arpit Singh Gautam` |
| Affiliation | Independent Researcher |

---

## Pre-submission checklist

Run through this **per submission**, immediately before uploading.

- [ ] **GitHub repo flipped to PUBLIC** and the URL opens in a private browser window
- [ ] PDF built from the official template, opens correctly, under 25 MB
- [ ] Abstract ≤ 150 words
- [ ] Author name and affiliation present
- [ ] Limitations and Dual-Use / Ethical Considerations appendix present
- [ ] Prior-work disclosure present (Aftermath declared; companion submission declared)
- [ ] Every link in the PDF resolves
- [ ] Titles are unique between the two submissions
- [ ] Publishing opt-in set to **Yes**
- [ ] Tracks selected as above
- [ ] Confirmation email received; if not within a few hours, email sprints@apartresearch.com

**Submit early.** Resubmitting under the exact same title replaces the files at no cost, so an
early safety submission removes deadline risk entirely.

---

## After submitting

A funding-opt-in form arrives by email. Ask **$20–50K**, against the roadmap in `decisions.md`:

1. Elicitation methods that work on weak instruction-followers and base checkpoints — our largest
   measured gap; Mistral was partly unmeasurable and the base checkpoint failed at its own baseline
2. Extension to valence and distress measurements
3. Extension to introspection probes with the same validity gating
4. Distributed / non-linear mechanistic interventions — the single-direction ablation null says
   this is needed
5. Pre-registration and replication infrastructure

The pitch is not "we found something about personas." It is: **four results in these two papers
would have been published wrong without diagnostics this field does not currently run.**
