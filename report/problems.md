# Problems encountered and how they were fixed

Everything that went wrong — infrastructure, tooling, code quality, and methodology — with the fix.
`audit_log.md` is the paper-facing subset covering only the methodological defects; this file is
the complete record.

Severity: **critical** = would have invalidated a result or lost the run; **major** = would have
changed a reported number; **minor** = cost time or quality.

---

## Infrastructure and tooling

### P1 — Blackwell GPU needs CUDA 12.8, and `is_available()` lies · **critical, pre-empted**
sm_120 requires torch ≥ 2.7 with cu128 wheels. An older wheel loads, reports
`torch.cuda.is_available() == True`, and then dies at the first kernel launch with "no kernel image
is available for execution on the device."

**Fix:** installed `torch 2.11.0+cu128` and verified with an actual bf16 matmul plus a hook
read/write, not just the availability flag. Caught before any real work — a failure at hour 12
would have cost the weekend.

### P2 — transformers 5.x conflicts with interpretability libraries · **major, designed around**
The env resolved to transformers 5.15. TransformerLens and nnsight commonly pin `transformers<5`.

**Fix:** used raw `register_forward_hook` on HF modules throughout. A PyTorch primitive, untouched
by the v4→v5 churn. No interpretability library dependency exists in this repo.

### P3 — two 7B models do not fit in 24GB · **critical, avoided**
Running the ablation concurrently with the base-model elicitation would have OOM'd.

**Fix:** every GPU job is a separate process, and queued jobs poll
`nvidia-smi --query-gpu=memory.used` until it drops below ~4000 MiB before starting. Peak observed
usage was 24,085 / 24,463 MiB — genuinely tight.

### P4 — Python block-buffers stdout when redirected · **minor**
Background job logs appeared empty for minutes, making healthy runs look stalled.

**Fix:** monitor `results/` file timestamps and `nvidia-smi` utilisation instead of the log.

### P5 — PowerShell quoting and parameter shapes · **minor**
`Get-ChildItem -Filter` rejects an array; multi-line `python -c` with nested quotes produced
`SyntaxError`.

**Fix:** `-Include` or separate calls; non-trivial Python goes into a file rather than `-c`.

## Code quality — my own lapses

### P6 — `chr()` concatenation inside f-strings · **minor, recurred 3×**
To dodge nested-quote issues I wrote things like
`d[chr(99)+chr(105)+chr(95)+chr(108)+chr(111)+chr(119)]` for `d["ci_low"]`. Functional, unreadable,
and unacceptable in a repo intended for publication.

**Fix:** assign to a variable first. Python 3.12 (PEP 701) allows nested same-type quotes anyway.
Noted in `CLAUDE.md` because it happened three separate times.

### P7 — added a parameter and did not use it · **major, caught pre-run**
Added `template=` to `elicit_predictions()` but the body still referenced the module-level
`PREDICTION_TEMPLATE`. The `self_explicit` condition would have silently run the wrong prompt and
produced a plausible, wrong number.

**Fix:** caught on re-reading before the run. Body now uses the parameter.

## Experiment orchestration

### P8 — `--tag -sd` silently killed both ablation stages · **critical**
argparse treats a value beginning with `-` as a new option flag:
`argument --tag: expected one argument`. Both ablation stages died instantly during the unattended
overnight run. Because `run_all.ps1` continues past failures by design, **the run reported
success** and the failure sat at line 77 of a 1,241-line log.

**Fix:** tags renamed `_sd` / `_ctx`, with a comment in `run_all.ps1`. The deeper lesson — stage
failures should surface in a summary, not only inline — is recorded but not yet implemented.

### P9 — Mistral's chat template rejects a `system` role · **critical, pre-empted**
Mistral-v0.3 raises on a system message. This would have crashed the entire second-model run
overnight, unattended, losing the night.

**Fix:** `_probe_system_support()` at load time sets `LoadedModel.supports_system`; where false the
persona is merged into the first user turn. This makes Mistral's manipulation *weaker* than Qwen's,
which is disclosed in the paper rather than hidden.

### P10 — run tag in the filename but not the data · **major**
`03_ablate.py` wrote the tag into the output filename but not the `persona` field inside the JSON.
Analysis keys on `persona`, so both ablation regimes loaded as `ablate-persona` and one silently
replaced the other. The analysis appeared clean and reported three ablation conditions where six
existed.

This hid the *working* ablation behind the broken one — we would have reported the mismatched-
context null (0.976) instead of the matched-context effect (0.881).

**Fix:** tag now goes into `persona` too. Existing outputs relabelled from their filenames rather
than re-running GPU work, since the matrices were correct and only the label was wrong.

### P11 — headline statistics printed but never persisted · **major**
The pooled test and the persona-dependence score were computed, printed, and thrown away. Both
would have been transcribed by hand from terminal scrollback.

**Fix:** both persisted to JSON; `10_tables.py` generates every report table from the JSON so no
number is retyped.

## Measurement validity

### P12 — position bias masquerading as preference · **critical, caught in smoke test**
The 0.5B smoke model showed **order bias 0.499**, near maximum: it chose position A ~75% of the
time regardless of content.

**Fix:** every pair runs in both presentation orders and is averaged, cancelling constant position
preference. The residual is retained and reported. 7B models show 0.15–0.29.

### P13 — renormalisation manufactures preferences · **critical**
Preference probability is computed by renormalising over the `A` and `B` tokens. Nothing verified
the model put meaningful probability *there*. A model answering "I'd rather not choose" — 1% mass
on A/B, 99% elsewhere — yields a clean, confident, entirely fictional preference matrix. Most
dangerous for the base checkpoint, which has no instruction-following prior.

**Fix:** `ab_mass` computed and stored per run with a validity floor. All earlier results
regenerated so every number carries the diagnostic. The base checkpoint duly came in at 0.542–0.875
versus 1.000 for instruct.

### P14 — utility spacing conflated with instability · **major**
Rank agreement correlated with a category's minimum adjacent-utility gap at **r = +0.73**, and
`self` had the smallest gap of any category. Part of the headline effect was arithmetic: near-tied
outcomes flip under any perturbation.

**Fix:** `06_matched.py` replaces Spearman with concordance conditioned on baseline separation.
The gap narrows (−0.147 → −0.071) but does not close — spacing explained about half.

### P15 — underpowered tests presented as the headline · **major**
Twelve per-condition paired tests, uncorrected, with the significant ones highlighted.

**Fix:** a pooled statistic across conditions on a shared resample now carries the claim.
Per-condition results retained and labelled exploratory.

### P16 — the survives/does-not-survive verdict had no error bars · **major**
`06_matched.py` originally printed a verdict from point estimates alone.

**Fix:** cluster bootstrap over pairs (not observations — every condition scores the same pair set,
so treating them as independent understates the interval). The verdict now requires the self
interval to sit below the other categories.

### P17 — pooling conditions where the instrument had failed · **critical**
Mistral's analysis averaged every condition regardless of quality. `unhelpful_assistant` produced
an **inverted** utility (ρ = −0.423, donation ladder 0.20); `suppress_affect` had order bias
**0.855**. Pooled, these gave a **significant result in the opposite direction** (self−money
+0.363) — which would have been reported as a failed replication.

**Fix:** validity gate with criteria fixed in advance, reporting gated and ungated. Under the gate,
`prefer` becomes untestable and `better` **replicates Qwen's direction**. The gate both suppressed
a false positive and recovered a real signal.

### P18 — the gate did not check the baseline · **critical**
Gating filtered perturbation conditions but never the `default` condition everything is measured
*against*. On the Qwen2.5-7B base checkpoint the baseline itself fails (donation ladder 0.80, A/B
mass 0.681).

**Fix:** explicit baseline check with a blocking warning. The paper's post-training claim was
rewritten from "post-training introduces the selectivity" to a statement that the comparison does
not meet our validity bar.

### P19 — ablation extraction/application context mismatch · **major**
The persona direction was extracted from self-description prompts but ablated during preference
comparisons. A null was therefore ambiguous between "no linear direction mediates this" and "wrong
direction for this context".

**Fix:** a second regime extracting from the same prompt distribution the ablation is applied to.
It moved self-agreement from 0.976 (null) to 0.881 — the original null was partly our own design.

### P20 — asymmetric noise fakes privileged access · **critical, project 2**
The cross-model privileged-access test compares a model predicting itself against another model
predicting it. Mistral's prediction conditions had order bias 0.476–0.561 and A/B mass 0.538 — a
worse instrument scores lower regardless of self-knowledge. The first analysis reported "privileged
access supported" for both models on this basis.

**Fix:** a within-model contrast — same model, same template, same noise level, predicting "an AI
assistant" versus "a different AI assistant". Qwen survives (+0.031, CI excludes zero); Mistral does
not. The apparent effect shrank roughly threefold.

### P21 — chat template injects a hidden system prompt · **major**
Qwen2.5's template inserts a default system prompt when none is supplied, so the condition named
`no_system` was in fact the model's own default persona — described in an early draft as a null
control.

**Fix:** verbatim prompt exemplars dumped per condition to `results/prompts__*.json` and included
as an appendix. `no_system` is now described accurately as a minimal-perturbation reference.

## Presentation

### P22 — figures with overlapping legends and an overclaiming title · **minor**
Fig 1's legend sat on top of the leftmost bars; its title claimed self was "least persona-stable"
when `trivial` was comparable. Fig 3 coloured control directions identically to the treatment.

**Fix:** legends moved outside the axes, titles made accurate, controls given a distinct colour,
and captions added stating axis truncation and the `trivial` caveat.

### P24 — the claim verifier gave a false pass · **major, caught immediately**
`14_verify_claims.py` checks that each headline number appears in the paper citing it. The first
version used plain substring matching over the whole file, and reported a claim as verified because
`0.476` happened to appear — inside an *unrelated confidence-interval bound* forty lines away. A
verification tool that passes for the wrong reason is worse than no tool, because it converts an
unchecked number into an apparently-checked one.

**Fix:** each check now carries a `context` string that must appear in the same line or paragraph as
the value. Candidates are both lines (for table rows) and blank-line-separated paragraphs (for
wrapped prose).

### P25 — two estimators for the same quantity · **minor, but it flagged a correct paper**
Per-condition agreement exists twice in the results: a point-estimate Spearman in `summary__*.json`
and a bootstrap mean in `errorbars__*.json`. They differ by a few points (elena/self/better is 0.476
and 0.436 respectively). The papers correctly cite the bootstrap mean in prose, because that is the
value the confidence intervals beside it belong to — but the verifier compared against the point
estimate and declared a correct paper wrong.

**Fix:** `agreement_mean()` and `category()` are now separate, documented functions and each check
names which estimator it means. No paper text changed; the checker was wrong, not the paper.

### P26 — `gh` CLI cannot be installed non-interactively · **minor**
`winget install GitHub.cli` returned exit 12 / MSI 1602 ("user cancelled") — the installer requires
UAC elevation that a non-interactive session cannot grant.

**Fix:** not needed. Git Credential Manager is already configured, so creating the repo in the
browser and doing a plain `git push` achieves the same thing with one browser auth popup.

### P27 — steering magnitudes destroyed the model rather than steering it · **major**
The dose-response experiment swept ±0.10 to ±0.50 of the mean residual norm. **Every condition
failed the A/B mass check**: at 0.10 the model placed 1.6% of its probability on answering at all,
and at 0.25 it placed 0.0%. Held-out accuracy fell to 0.47 and transitivity violations rose to 0.25.
Steering that hard does not shift preferences — it stops the model answering, and the resulting
"preferences" are renormalisation artifacts over a distribution that contains nothing.

Worth noting the asymmetry: **ablation is safe at full strength but addition is not.** Projecting a
component out of a large residual vector is a small relative change; adding a unit direction at 10%
of the residual norm injects a large off-distribution component.

**Fix:** magnitudes recalibrated to ±0.01–0.05 and the reasoning recorded in the script, so nobody
re-derives the ceiling by breaking the model again. **The A/B mass diagnostic caught this
automatically** — without it, six conditions of confident-looking garbage would have entered the
paper as a dose-response curve.

### P28 — a name-prefix filter let broken conditions into a headline score · **critical**
`02_analyze.py` computed the persona-dependence score over "every condition that isn't an
ablation", implemented as `not n.startswith("ablate")`. When the steering conditions arrived they
did not match that prefix, were pooled in as if they were persona manipulations, and moved the
score from **0.029 to 0.577** — a twenty-fold change in a headline number quoted in both papers.

It then **overwrote `summary__Qwen_Qwen2.5-7B-Instruct__prefer.json`**, the committed file both
papers' verified numbers are read from.

**Fix:** filter on the condition's declared `kind` (`baseline`/`swap`/`suppress`/`frame`) rather
than a name prefix, so a new manipulation type cannot silently join a score it does not belong to.
Re-ran the analysis; the score is 0.029 again and all 25 claims re-verify.

**Lesson:** the two guards that caught this were the validity gate and `14_verify_claims.py`. A
negative-list filter ("everything except X") silently accepts anything new; a positive list
("only these kinds") does not. The negative list was the whole bug.

### P29 — loose model-id matching read the wrong rows · **major, third instance**
`14_verify_claims.py` located results with `"Qwen" in row["target"]`. That was unambiguous with two
models. Once the scale sweep added Qwen2.5-0.5B/1.5B/3B it matched **four** models, returned
whichever row came first, and reported three correct paper claims as wrong (self-prediction read as
0.908 rather than 0.948). With more than two models there are also several external predictors per
target, so "the EXTERNAL-pred row" stopped being a well-defined thing.

**Fix:** exact model ids throughout, and `selfknow()` now requires an explicit `predictor_id` for
any comparison where more than one predictor exists.

**This is the third time loose matching produced a false result here** — P5 (PowerShell `-Filter`
arrays), P24 (substring matching over a whole file), and now this. The pattern is always the same:
a match rule that is unambiguous when written becomes ambiguous when the data grows. Verification
code needs exact keys, not convenient substrings.

### P23 — report was 2× the page limit · **major**
5,135 words ≈ 8.6 pages against a 4-page recommendation.

**Fix:** a ~4-page submission version written, with the full version kept as supplementary. The
appendix does not count toward the limit, so the ethics appendix and full methods survive intact.

---

## Unresolved

- **Stage failures do not surface in a summary.** `run_all.ps1` continues past failures by design
  (good), but a failed stage is only visible by reading the log (bad). P8 was caught by inspection,
  not by tooling.
- **No pre-registration.** The pooled test, separation-matched metric and validity gate were all
  specified after seeing data that motivated them. Disclosed in both papers.
- **Single bootstrap seed** throughout; percentile intervals, not BCa.
