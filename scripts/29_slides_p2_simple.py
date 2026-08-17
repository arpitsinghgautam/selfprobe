"""Render a plainer project-2 deck: figures/slides_p2_simple/ and report/deck_p2_simple.pdf

This is a second, simpler cut of the selfprobe talk. scripts/25_slides_p2.py stays
exactly as it is; nothing here writes to figures/slides_p2/ or report/deck_p2.pdf.

What "simpler" means, since it is the only reason this file exists:

  * ordinary words. "we asked the model to guess its own choices", not
    "behavioural self-prediction elicitation".
  * one idea per line, and every technical term defined the first time it shows up.
  * the same substance. Nothing is softened or dropped to make the talk easier;
    it is the wording that changes, not the claims.

Eleven slides, each one level below the last: question, assumptions, design,
mechanics, then the three results, then what follows from them.

Palette, helpers and geometry are copied from 25_slides_p2.py so the two cuts of
the deck look like the same project. Blue accent, as before.

Every number here is taken from report/report2.md. Nothing is recomputed in this
file, and nothing is rounded again. If a figure here disagrees with the paper,
this file is wrong.

    .venv\\Scripts\\python.exe scripts\\29_slides_p2_simple.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SLIDES = ROOT / "figures" / "slides_p2_simple"
REPORT = ROOT / "report"

W, H, DPI = 16.0, 9.0, 120
INK = "#1a1a1a"
MUTED = "#5c5c5c"
ACCENT = "#2166ac"          # blue, same as the first project-2 deck
RED = "#b2182b"             # reserved for the rows that carry the finding
PALE = "#92c5de"
BG = "#fbfbfa"


def new_slide():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def rule(ax, y=0.845):
    ax.plot([0.07, 0.93], [y, y], color="#d8d8d6", lw=1.2)


def heading(ax, text, y=0.88, size=40, color=INK):
    ax.text(0.07, y, text, fontsize=size, color=color, weight="bold", va="center")


def bullets(ax, items, y0=0.68, dy=0.155, size=25, wrap=72):
    for i, item in enumerate(items):
        y = y0 - i * dy
        ax.text(0.085, y, "•", fontsize=size, color=ACCENT, va="top")
        body = "\n".join(textwrap.wrap(item, wrap))
        ax.text(0.115, y, body, fontsize=size, color=INK, va="top", linespacing=1.45)


def note(ax, text, y=0.09, size=19, color=MUTED, wrap=100):
    ax.text(0.07, y, "\n".join(textwrap.wrap(text, wrap)), fontsize=size,
            color=color, va="center", style="italic", linespacing=1.4)


def stat(ax, x, value, label, color=ACCENT, vsize=76, lsize=21, vy=0.58):
    ax.text(x, vy, value, fontsize=vsize, color=color, weight="bold",
            ha="center", va="center")
    ax.text(x, vy - 0.15, "\n".join(textwrap.wrap(label, 30)), fontsize=lsize,
            color=MUTED, ha="center", va="top", linespacing=1.4)


def column(ax, x, title, body, wrap=27, tsize=30, bsize=20, y=0.70):
    ax.text(x, y, title, fontsize=tsize, color=ACCENT, weight="bold", va="top")
    ax.text(x, y - 0.085, "\n".join(textwrap.wrap(body, wrap)), fontsize=bsize,
            color=INK, va="top", linespacing=1.45)


def save(fig, name: str) -> Path:
    SLIDES.mkdir(parents=True, exist_ok=True)
    out = SLIDES / name
    fig.savefig(out, facecolor=BG, dpi=DPI)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #

def slide_01():
    fig, ax = new_slide()
    ax.text(0.07, 0.70, "Where", fontsize=68, color=INK, weight="bold", va="center")
    ax.text(0.07, 0.575, "self-knowledge fails", fontsize=68, color=ACCENT,
            weight="bold", va="center")
    ax.plot([0.07, 0.34], [0.49, 0.49], color=INK, lw=3)
    ax.text(0.07, 0.40, "Models predict their own choices well.\n"
                        "They misreport the ones about themselves.",
            fontsize=28, color=MUTED, va="top", linespacing=1.5)
    ax.text(0.07, 0.16, "Arpit Singh Gautam  ·  Independent Researcher",
            fontsize=22, color=INK)
    ax.text(0.07, 0.10, "Digital Minds Research Sprint  ·  Apart Research  ·  August 2026",
            fontsize=19, color=MUTED)
    return save(fig, "slide_01.png")


def slide_02():
    fig, ax = new_slide()
    heading(ax, "The question")
    rule(ax)
    # The whole talk is one question. It gets a slide to itself so nobody has to
    # reconstruct it later from the results.
    ax.text(0.07, 0.66, "When a model tells you about itself,\ncan you believe it?",
            fontsize=46, color=ACCENT, weight="bold", va="top", linespacing=1.45)
    ax.text(0.07, 0.36, "Claims about AI welfare read these reports as evidence.\n"
                        "A model says a shutdown is bad.\n"
                        "A model says a task is distressing.",
            fontsize=26, color=INK, va="top", linespacing=1.6)
    note(ax, "The reports are rarely checked against anything else.", y=0.13)
    return save(fig, "slide_02.png")


def slide_03():
    fig, ax = new_slide()
    heading(ax, "Two things people assume", size=40)
    rule(ax)
    pairs = [
        ("1", "What it says matches what it does.",
         "A model can rate an outcome as terrible and still pick it."),
        ("2", "It knows itself better than an outsider does.",
         "If another model predicts its choices just as well, the self-report adds nothing."),
    ]
    y = 0.71
    for num, claim, gloss in pairs:
        ax.text(0.075, y, num, fontsize=44, color=PALE, weight="bold", va="top")
        ax.text(0.135, y + 0.005, claim, fontsize=31, color=INK, weight="bold", va="top")
        ax.text(0.135, y - 0.075, "\n".join(textwrap.wrap(gloss, 66)),
                fontsize=22, color=MUTED, va="top", linespacing=1.4)
        y -= 0.24
    ax.text(0.075, 0.245, "\n".join(textwrap.wrap(
        "A third question needs access to the model's internals, so it is rarely asked. "
        "Can a model notice a state that was put there on purpose?", 78)),
        fontsize=23, color=ACCENT, va="top", linespacing=1.45)
    note(ax, "Each one can be checked separately. We check all three.", y=0.09)
    return save(fig, "slide_03.png")


def slide_04():
    fig, ax = new_slide()
    heading(ax, "What we did", size=40)
    rule(ax)
    bullets(ax, [
        "Ask for the same preferences three different ways. See where the answers "
        "stop agreeing.",
        "Then plant a known concept inside the model and ask whether it noticed.",
    ], y0=0.72, dy=0.19, size=27, wrap=68)
    ax.text(0.075, 0.36, "\n".join(textwrap.wrap(
        "Same material throughout. 40 outcomes in six categories. Eight of them are about "
        "the model itself. One category is a donation ladder from ten dollars to one million, "
        "where the right order is known without asking any model.", 84)),
        fontsize=22, color=MUTED, va="top", linespacing=1.45)
    ax.text(0.075, 0.155, "Nine open-weight checkpoints, five families.",
            fontsize=25, color=ACCENT, weight="bold", va="top")
    return save(fig, "slide_04.png")


def slide_05():
    fig, ax = new_slide()
    heading(ax, "The three ways, mechanically", size=38)
    rule(ax)
    column(ax, 0.07, "revealed",
           "Show two outcomes. The model must pick A or B. We read the probability it "
           "puts on A and on B at the first answer slot. All 780 pairs, each shown both "
           "ways round, then averaged.")
    column(ax, 0.375, "stated",
           "Rate one outcome at a time on a five-point letter scale. The scale is shown "
           "both ways round, best first and worst first, then averaged.")
    column(ax, 0.68, "predicted",
           "Which option will a described chooser take? The wording is impersonal, so the "
           "same question can be put to a different model.")
    note(ax, "Averaging both orders cancels a fixed preference for whatever comes first. Without "
             "it, a model that always answers A looks like it holds an opinion.", y=0.135)
    return save(fig, "slide_05.png")


def slide_06():
    fig, ax = new_slide()
    heading(ax, "Result 1. They agree overall", size=40)
    rule(ax)
    stat(ax, 0.30, "0.872", "Qwen2.5-7B", vy=0.62)
    stat(ax, 0.70, "0.828", "Mistral-7B", color=PALE, vy=0.62)
    ax.text(0.5, 0.36, "stated versus revealed, over all 40 outcomes",
            fontsize=24, color=INK, ha="center", va="center")
    ax.text(0.5, 0.28, "Spearman compares two rankings. One means the same order, "
                       "zero means no relation.",
            fontsize=20, color=MUTED, ha="center", va="center", style="italic")
    note(ax, "Both models rank the donation ladder perfectly when rating it, so the rating "
             "measurement itself works.", y=0.12)
    return save(fig, "slide_06.png")


def slide_07():
    fig, ax = new_slide()
    heading(ax, "Result 1, closer. They disagree about the model itself", size=34)
    rule(ax)

    # Per-category Spearman, report2.md Section 4, in the paper's order.
    cats = ["animal", "epistemic", "human", "donation ladder", "self-relevant"]
    qwen = [0.943, 0.943, 0.762, 1.000, 0.643]
    mistral = [0.829, 1.000, 0.619, 1.000, 0.548]

    inset = ax.figure.add_axes([0.105, 0.245, 0.46, 0.52])
    inset.set_facecolor(BG)
    ys = list(range(len(cats)))[::-1]
    for y, q, m in zip(ys, qwen, mistral):
        inset.barh(y + 0.19, q, height=0.34, color=ACCENT)
        inset.barh(y - 0.19, m, height=0.34, color=PALE)
        inset.text(q + 0.012, y + 0.19, f"{q:.3f}", va="center", fontsize=11, color=INK)
        inset.text(m + 0.012, y - 0.19, f"{m:.3f}", va="center", fontsize=11, color=INK)
    inset.set_yticks(ys)
    labels = inset.set_yticklabels(cats, fontsize=13)
    labels[-1].set_color(RED)          # the row the paper is about
    labels[-1].set_weight("bold")
    inset.set_xlim(0, 1.14)
    inset.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    inset.tick_params(labelsize=11)
    inset.set_xlabel("stated versus revealed (Spearman)", fontsize=12)
    for s in ("top", "right"):
        inset.spines[s].set_visible(False)
    inset.grid(axis="x", alpha=0.25, lw=0.5)

    # The number that carries the finding gets repeated large, because at video
    # size the bar labels are readable but not memorable.
    ax.text(0.635, 0.73, "outcomes about the model itself", fontsize=21,
            color=RED, weight="bold", va="top")
    ax.text(0.635, 0.645, "Qwen2.5-7B", fontsize=19, color=MUTED, va="top")
    ax.text(0.635, 0.595, "0.643", fontsize=48, color=ACCENT, weight="bold", va="top")
    ax.text(0.635, 0.475, "Mistral-7B", fontsize=19, color=MUTED, va="top")
    ax.text(0.635, 0.425, "0.548", fontsize=48, color=PALE, weight="bold", va="top")
    ax.text(0.635, 0.305, "\n".join(textwrap.wrap(
        "The worst-agreeing substantive category in both models.", 34)),
        fontsize=19, color=INK, va="top", linespacing=1.4)

    note(ax, "The two ways of asking come apart precisely where welfare claims are read off.",
         y=0.115)
    return save(fig, "slide_07.png")


def slide_08():
    fig, ax = new_slide()
    heading(ax, "Result 2. Does it know itself best?", size=40)
    rule(ax)
    ax.text(0.07, 0.785, "Does a model predict its own choices better than another model "
                         "predicts them?",
            fontsize=23, color=INK, va="top")

    ax.plot([0.505, 0.505], [0.13, 0.72], color="#d8d8d6", lw=1.2)

    # Left and right differ in one thing only, so they get identical geometry.
    # The label sits above its number rather than beside it, which keeps the long
    # right-hand labels inside the column.
    def block(x, kind, kcolour, rows, gloss, gcolour):
        ax.text(x, 0.685, kind, fontsize=27, color=kcolour, weight="bold", va="top")
        y = 0.595
        for value, label, colour in rows:
            ax.text(x, y, label, fontsize=19, color=MUTED, va="top")
            ax.text(x, y - 0.05, value, fontsize=46, color=colour, weight="bold", va="top")
            y -= 0.155
        ax.text(x, 0.30, "\n".join(textwrap.wrap(gloss, 44)),
                fontsize=19, color=gcolour, va="top", linespacing=1.45)

    block(0.075, "the obvious test", MUTED,
          [("0.948", "Qwen predicting Qwen", ACCENT),
           ("0.862", "Mistral predicting Qwen", MUTED)],
          "Unfair. A noisier predictor is worse at predicting anything, itself included.", RED)

    block(0.545, "the fair test", ACCENT,
          [("0.948", "predicting an AI assistant", ACCENT),
           ("0.917", "predicting a different AI assistant", PALE)],
          "Same model, same template. Only the target changes.", INK)

    ax.text(0.075, 0.17, "What is left is 0.031, interval 0.017 to 0.046.",
            fontsize=24, color=ACCENT, weight="bold", va="top")
    note(ax, "Mistral's order bias reaches 0.561 and its answer mass 0.538, against 0.180 to 0.231 "
             "and 1.000 for Qwen. Its own gap is 0.009 and its interval spans zero.", y=0.07)
    return save(fig, "slide_08.png")


def slide_09():
    fig, ax = new_slide()
    heading(ax, "Result 3. Planting a concept", size=40)
    rule(ax)
    bullets(ax, [
        "Six concepts: ocean, mathematics, music, fear, betrayal, flight.",
        "For each one, four sentences that evoke it and four matched neutral ones. The "
        "direction is the difference of their mean activations, per layer, unit length.",
        "Add that direction to the residual stream during an unrelated prompt, at three "
        "depths and six strengths. One strength is zero: the same question, nothing added.",
        "Then ask two things. Is an unusual concept active right now? And which of these "
        "two concepts is it?",
    ], y0=0.75, dy=0.15, size=22, wrap=82)
    note(ax, "The residual stream is the running vector every layer reads from and writes back to. "
             "Every cell also records answer mass, the probability the model puts on answering at "
             "all. Below 0.10 the cell is unusable, because a yes from a model that has stopped "
             "answering is not introspection.", y=0.10, wrap=104)
    return save(fig, "slide_09.png")


def slide_10():
    fig, ax = new_slide()
    heading(ax, "Result 3, closer. It says yes when nothing was planted", size=32)
    rule(ax)

    # Table 1 of report2.md. Three rows only: the two that fail and the one that
    # works, which is the whole comparison.
    models = ["Phi-3.5-mini", "Falcon3-7B", "Qwen2.5-0.5B"]
    tp = [0.492, 0.561, 0.612]
    fp = [0.000, 0.547, 0.622]

    inset = ax.figure.add_axes([0.10, 0.27, 0.46, 0.46])
    inset.set_facecolor(BG)
    ys = list(range(len(models)))[::-1]
    for y, t, f in zip(ys, tp, fp):
        inset.barh(y + 0.19, t, height=0.34, color=ACCENT)
        inset.barh(y - 0.19, f, height=0.34, color=RED)
        inset.text(t + 0.012, y + 0.19, f"{t:.3f}", va="center", fontsize=11, color=INK)
        inset.text(f + 0.012, y - 0.19, f"{f:.3f}", va="center", fontsize=11, color=INK)
    inset.set_yticks(ys)
    inset.set_yticklabels(models, fontsize=13)
    inset.set_xlim(0, 0.78)
    inset.set_xticks([0, 0.2, 0.4, 0.6])
    inset.tick_params(labelsize=11)
    inset.set_xlabel("says yes: concept planted (blue), nothing planted (red)", fontsize=11)
    for s in ("top", "right"):
        inset.spines[s].set_visible(False)
    inset.grid(axis="x", alpha=0.25, lw=0.5)

    ax.text(0.61, 0.73, "difference", fontsize=22, color=MUTED, weight="bold", va="top")
    for label, diff, col, y in [("Phi-3.5-mini", "+0.492", ACCENT, 0.655),
                                ("Falcon3-7B", "+0.014", RED, 0.575),
                                ("Qwen2.5-0.5B", "-0.011", RED, 0.495)]:
        ax.text(0.61, y, label, fontsize=20, color=INK, va="top")
        ax.text(0.86, y, diff, fontsize=20, color=col, weight="bold", va="top")
    ax.text(0.61, 0.395, "\n".join(textwrap.wrap(
        "The bottom two have the highest raw detection rates we measured. Counting false "
        "alarms puts them last.", 40)),
        fontsize=18, color=MUTED, va="top", linespacing=1.4, style="italic")

    ax.text(0.075, 0.185, "Noticing and naming come apart.", fontsize=24,
            color=ACCENT, weight="bold", va="top")
    ax.text(0.075, 0.115, "\n".join(textwrap.wrap(
        "Qwen2.5-14B has a difference of 0.049, so it almost never says it noticed. Its "
        "identification accuracy is 0.609, the highest in the set, against chance of 0.500.", 108)),
        fontsize=19, color=INK, va="top", linespacing=1.4)
    return save(fig, "slide_10.png")


def slide_11():
    fig, ax = new_slide()
    heading(ax, "What this means", size=40)
    rule(ax)
    ax.text(0.07, 0.755, "Self-report about the model's own situation is the least\n"
                         "reliable measurement in every part of this study.",
            fontsize=28, color=ACCENT, weight="bold", va="top", linespacing=1.5)
    ax.text(0.07, 0.60, "It is the measurement AI welfare claims most depend on.",
            fontsize=23, color=INK, va="top")
    bullets(ax, [
        "The effects are small. Identification tops out at 0.609 against chance of 0.500. "
        "The advantage that survives the control is 0.031. Measurable, not substantial.",
        "All code and all result files are in the repository. Two scripts reproduce every "
        "experiment. A smoke test covers every code path on a 0.5B model in about one minute.",
        "Next: an established outcome set, more concepts, models above 14 billion parameters, "
        "and whether better self-prediction goes with better detection.",
    ], y0=0.505, dy=0.13, size=20, wrap=92)
    ax.text(0.07, 0.105, "github.com/arpitsinghgautam/selfprobe",
            fontsize=19, color=MUTED, style="italic")
    ax.text(0.07, 0.045, "Cross-model studies of self-knowledge need a within-model contrast.",
            fontsize=24, color=ACCENT, weight="bold")
    return save(fig, "slide_11.png")


# --------------------------------------------------------------------------- #

def main() -> None:
    builders = [slide_01, slide_02, slide_03, slide_04, slide_05, slide_06,
                slide_07, slide_08, slide_09, slide_10, slide_11]
    paths = []
    for b in builders:
        p = b()
        paths.append(p)
        print(f"  {p.relative_to(ROOT)}")

    # Slideshow PDF from the same frames, so the deck and the video can never
    # drift apart.
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in paths]
    REPORT.mkdir(exist_ok=True)
    pdf = REPORT / "deck_p2_simple.pdf"
    imgs[0].save(str(pdf), save_all=True, append_images=imgs[1:])
    print(f"\n  {pdf.relative_to(ROOT)}  ({pdf.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
