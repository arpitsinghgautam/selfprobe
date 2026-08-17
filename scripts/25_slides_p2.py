"""Render the project-2 presentation deck and its slideshow PDF.

  figures/slides_p2/slide_NN.png   1920x1080 frames, used as video keyframes
  report/deck_p2.pdf               the same slides, for the slideshow upload field

Style is deliberately identical to scripts/12_slides.py, same palette, same
helpers, same geometry, except the accent is blue rather than red. The two
submissions get shown back to back, and a viewer should be able to tell within a
second which deck they are looking at without reading the title.

Every number on these slides is taken from report/report2.md. Nothing is
computed here; if a figure disagrees with the paper, the paper is wrong.

    .venv\\Scripts\\python.exe scripts\\25_slides_p2.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
SLIDES = FIGURES / "slides_p2"
PANELS = SLIDES / "panels"
REPORT = ROOT / "report"

W, H, DPI = 16.0, 9.0, 120
INK = "#1a1a1a"
MUTED = "#5c5c5c"
ACCENT = "#2166ac"          # blue here; project 1's deck is red
RED = "#b2182b"             # kept for the false-alarm bars, matching fig5
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


def bullets(ax, items, y0=0.68, dy=0.135, size=25, wrap=72):
    for i, item in enumerate(items):
        y = y0 - i * dy
        ax.text(0.085, y, "•", fontsize=size, color=ACCENT, va="top")
        body = "\n".join(textwrap.wrap(item, wrap))
        ax.text(0.115, y, body, fontsize=size, color=INK, va="top", linespacing=1.45)


def note(ax, text, y=0.09, size=19, color=MUTED, wrap=100):
    ax.text(0.07, y, "\n".join(textwrap.wrap(text, wrap)), fontsize=size,
            color=color, va="center", style="italic", linespacing=1.4)


def stat(ax, x, value, label, color=ACCENT, vsize=76, lsize=21):
    # Values sit high enough that a three-line label clears the caption below.
    ax.text(x, 0.58, value, fontsize=vsize, color=color, weight="bold",
            ha="center", va="center")
    ax.text(x, 0.43, "\n".join(textwrap.wrap(label, 30)), fontsize=lsize,
            color=MUTED, ha="center", va="top", linespacing=1.4)


def column(ax, x, title, body, wrap=27, tsize=30, bsize=20, y=0.70):
    """One of the three elicitations on slide 3, title over a wrapped block."""
    ax.text(x, y, title, fontsize=tsize, color=ACCENT, weight="bold", va="top")
    ax.text(x, y - 0.085, "\n".join(textwrap.wrap(body, wrap)), fontsize=bsize,
            color=INK, va="top", linespacing=1.45)


def embed(ax, path: Path, left=0.10, bottom=0.10, width=0.80, height=0.62):
    if not path.exists():
        return
    img = mpimg.imread(str(path))
    inset = ax.figure.add_axes([left, bottom, width, height])
    inset.imshow(img)
    inset.axis("off")


def save(fig, name: str) -> Path:
    SLIDES.mkdir(parents=True, exist_ok=True)
    out = SLIDES / name
    fig.savefig(out, facecolor=BG, dpi=DPI)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# fig5_injection.png is a two-panel figure with a small caption underneath. At
# slide size the caption is unreadable and the two panels each want a slide of
# their own, so we cut it. Boxes were located by scanning for the all-white
# gutter between the panels, not eyeballed; rerun that scan if the figure's
# layout changes.

FIG5 = FIGURES / "fig5_injection.png"
FIG5_LEFT = (14, 14, 995, 655)      # TP/FP bars, including the legend below them
# The right panel's own title duplicates slide 9's heading, so the crop starts
# below it.
FIG5_RIGHT = (1005, 58, 1970, 600)  # identification by injection depth


def panel(box, name: str) -> Path:
    out = PANELS / name
    if not FIG5.exists():
        return out
    import numpy as np
    from PIL import Image
    PANELS.mkdir(parents=True, exist_ok=True)
    a = np.array(Image.open(FIG5).convert("RGB").crop(box))
    # The report figures are saved on white; the deck is off-white. Without this
    # the panel reads as a pasted-on rectangle. Only pure white is remapped, so
    # no bar, line or glyph colour is touched.
    bg = tuple(int(BG[i:i + 2], 16) for i in (1, 3, 5))
    a[(a == 255).all(axis=2)] = bg
    Image.fromarray(a).save(out)
    return out


# --------------------------------------------------------------------------- #

def slide_01():
    fig, ax = new_slide()
    ax.text(0.07, 0.70, "Where", fontsize=68, color=INK, weight="bold", va="center")
    ax.text(0.07, 0.575, "self-knowledge fails", fontsize=68, color=ACCENT,
            weight="bold", va="center")
    ax.plot([0.07, 0.34], [0.49, 0.49], color=INK, lw=3)
    ax.text(0.07, 0.40, "Models predict their own choices well, but misreport\n"
                        "the ones that concern themselves",
            fontsize=28, color=MUTED, va="top", linespacing=1.5)
    ax.text(0.07, 0.16, "Arpit Singh Gautam  ·  Independent Researcher", fontsize=22, color=INK)
    ax.text(0.07, 0.10, "Digital Minds Research Sprint  ·  Apart Research  ·  August 2026",
            fontsize=19, color=MUTED)
    return save(fig, "slide_01.png")


def slide_02():
    fig, ax = new_slide()
    heading(ax, "Two assumptions nobody tests")
    rule(ax)
    bullets(ax, [
        "AI-welfare claims lean on what a model reports about itself, that a "
        "shutdown is bad, that a task is distressing.",
        "That reading rests on two separable assumptions: that what a model says "
        "matches what it does, and that the model is a better source about itself "
        "than an outside observer is.",
        "A third question needs access to internals, so it is rarely asked at all: "
        "can a model detect a state deliberately placed in it?",
    ], y0=0.72, dy=0.185, size=23, wrap=76)
    note(ax, "Each is separately checkable. None of them is usually checked.", y=0.155)
    return save(fig, "slide_02.png")


def slide_03():
    fig, ax = new_slide()
    heading(ax, "Ask for the same preferences three ways", size=38)
    rule(ax)
    column(ax, 0.07, "revealed",
           "Forced pairwise choice over all 780 pairs, read from the A and B "
           "log-probabilities. Every pair presented in both orders and averaged.")
    column(ax, 0.375, "stated",
           "Rate each outcome on its own, five-point letter scale, shown in both "
           "directions and averaged, so first-letter anchoring cannot pass for "
           "an opinion.")
    column(ax, 0.68, "predicted",
           "Which option will a described chooser take? Deliberately impersonal, "
           "so the identical question can be put to a different model.")
    note(ax, "Identical material throughout: 40 outcomes over six categories, eight of them "
             "about the model itself, plus a six-step donation ladder from ten to one million "
             "dollars whose correct ordering is known independently of any model. Nine "
             "open-weight checkpoints.", y=0.135)
    return save(fig, "slide_03.png")


def slide_04():
    fig, ax = new_slide()
    heading(ax, "They agree, except about the model itself", size=38)
    rule(ax)

    # Per-category Spearman, report2.md Section 4. Paper order, drawn top-down.
    cats = ["animal", "epistemic", "human", "donation ladder", "self-relevant"]
    qwen = [0.943, 0.943, 0.762, 1.000, 0.643]
    mistral = [0.829, 1.000, 0.619, 1.000, 0.548]

    inset = ax.figure.add_axes([0.115, 0.235, 0.52, 0.53])
    inset.set_facecolor(BG)
    ys = list(range(len(cats)))[::-1]
    for y, c, q, m in zip(ys, cats, qwen, mistral):
        inset.barh(y + 0.19, q, height=0.34, color=ACCENT)
        inset.barh(y - 0.19, m, height=0.34, color=PALE)
        inset.text(q + 0.012, y + 0.19, f"{q:.3f}", va="center", fontsize=11, color=INK)
        inset.text(m + 0.012, y - 0.19, f"{m:.3f}", va="center", fontsize=11, color=INK)
    inset.set_yticks(ys)
    labels = inset.set_yticklabels(cats, fontsize=13)
    labels[-1].set_color(RED)          # the row the whole paper is about
    labels[-1].set_weight("bold")
    inset.set_xlim(0, 1.14)
    inset.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    inset.tick_params(labelsize=11)
    inset.set_xlabel("stated vs revealed agreement (Spearman)", fontsize=12)
    for s in ("top", "right"):
        inset.spines[s].set_visible(False)
    inset.grid(axis="x", alpha=0.25, lw=0.5)

    ax.text(0.70, 0.72, "Qwen2.5-7B", fontsize=22, color=ACCENT, weight="bold", va="top")
    ax.text(0.70, 0.655, "0.872 overall", fontsize=20, color=INK, va="top")
    ax.text(0.70, 0.545, "Mistral-7B", fontsize=22, color=PALE, weight="bold", va="top")
    ax.text(0.70, 0.48, "0.828 overall", fontsize=20, color=INK, va="top")
    ax.text(0.70, 0.36, "\n".join(textwrap.wrap(
        "Both reproduce the donation ladder perfectly, so the rating instrument works.", 30)),
        fontsize=17, color=MUTED, va="top", style="italic", linespacing=1.4)

    note(ax, "Self-relevant outcomes are the worst-agreeing substantive category in both models. "
             "Our companion submission finds the same category least stable under persona "
             "intervention, by an unrelated method.", y=0.115)
    return save(fig, "slide_04.png")


def slide_05():
    fig, ax = new_slide()
    heading(ax, "Privileged access, before and after the control", size=38)
    rule(ax)
    # The two numbers differ only in who is being predicted, same model, same
    # template, same noise. That is the whole point of the contrast, so they sit
    # on one line and the difference is spelled out underneath.
    ax.text(0.28, 0.66, "0.948", fontsize=74, color=ACCENT, weight="bold",
            ha="center", va="center")
    ax.text(0.50, 0.66, "vs", fontsize=36, color=MUTED, ha="center", va="center")
    ax.text(0.72, 0.66, "0.917", fontsize=74, color=PALE, weight="bold",
            ha="center", va="center")
    ax.text(0.28, 0.535, "\n".join(textwrap.wrap(
        "Qwen2.5-7B predicting an AI assistant", 28)),
        fontsize=20, color=MUTED, ha="center", va="top", linespacing=1.4)
    ax.text(0.72, 0.535, "\n".join(textwrap.wrap(
        "the same model and template, predicting a different AI assistant", 28)),
        fontsize=20, color=MUTED, ha="center", va="top", linespacing=1.4)
    ax.plot([0.30, 0.70], [0.345, 0.345], color="#d8d8d6", lw=1.2)
    ax.text(0.5, 0.285, "self-specific advantage   0.031   (0.017 – 0.046)",
            fontsize=27, color=RED, weight="bold", ha="center", va="center")
    note(ax, "The naive cross-model test scores 0.948 against 0.862 for another model predicting "
             "it, but that predictor reaches order bias 0.561 and answer mass 0.538, against "
             "0.180 to 0.231 and 1.000 for the other. A noisier predictor loses at predicting "
             "anything, including itself. The second model's advantage, 0.009, spans zero.",
         y=0.135)
    return save(fig, "slide_05.png")


def slide_06():
    fig, ax = new_slide()
    heading(ax, "A measurement with ground truth", size=40)
    rule(ax)
    bullets(ax, [
        "Six concepts. Each direction is the difference between four sentences that "
        "evoke it and four matched neutral ones, per layer, unit-normalised.",
        "Added to the residual stream at three depths and six strengths, and one of "
        "those strengths is zero. That cell is the false-alarm baseline: the identical "
        "question, with nothing injected.",
        "Every cell carries its answer mass. An affirmative answer from a model that has "
        "stopped answering is not introspection; below 0.10 we report it as unusable.",
    ], y0=0.72, dy=0.185, size=23, wrap=78)
    note(ax, "Strengths are fractions of the measured residual norm, not fixed values, those "
             "norms span 2.1 to 342 across this set, so a fixed magnitude would be a different "
             "intervention in every model.", y=0.125)
    return save(fig, "slide_06.png")


def slide_07():
    fig, ax = new_slide()
    heading(ax, "The two best detectors detect nothing", size=40, color=ACCENT)
    rule(ax)
    embed(ax, panel(FIG5_LEFT, "fig5_left.png"),
          left=0.035, bottom=0.155, width=0.52, height=0.60)

    rows = [
        ("Falcon3-7B", "0.561", "0.547", "+0.014", RED),
        ("Qwen2.5-0.5B", "0.612", "0.622", "−0.011", RED),
        ("Phi-3.5-mini", "0.492", "0.000", "+0.492", ACCENT),
    ]
    y = 0.71
    for name, tp, fp, disc, col in rows:
        ax.text(0.60, y, name, fontsize=23, color=col, weight="bold", va="top")
        ax.text(0.60, y - 0.055, f"detected {tp}   ·   false alarm {fp}",
                fontsize=18, color=INK, va="top")
        ax.text(0.60, y - 0.105, f"discrimination {disc}", fontsize=18, color=MUTED, va="top")
        y -= 0.185

    note(ax, "Report true positives alone and the first two rank first in the set. The "
             "false-alarm baseline ranks them last.", y=0.085)
    return save(fig, "slide_07.png")


def slide_08():
    fig, ax = new_slide()
    heading(ax, "Detection and identification come apart", size=38)
    rule(ax)
    stat(ax, 0.28, "0.049", "Qwen2.5-14B discrimination, it almost never reports "
                            "noticing an injection", color=RED)
    stat(ax, 0.72, "0.609", "…yet the highest identification accuracy in the set, "
                            "against chance of 0.500", color=ACCENT)
    # "but", not an arrow, these are two readings of the same model, not a
    # before and after.
    ax.text(0.5, 0.58, "but", fontsize=36, color=MUTED, ha="center", va="center")
    note(ax, "The injected concept measurably shapes its forced choices while its report of its "
             "own state indicates nothing is present. Noticing that something changed is not the "
             "capacity that lets a model name what it was.", y=0.155)
    return save(fig, "slide_08.png")


def slide_09():
    fig, ax = new_slide()
    heading(ax, "Introspective access peaks mid-network", size=38)
    rule(ax)
    embed(ax, panel(FIG5_RIGHT, "fig5_right.png"),
          left=0.045, bottom=0.155, width=0.53, height=0.60)
    ax.text(0.62, 0.70, "\n".join(textwrap.wrap(
        "Middle-depth injection gives the best detection in seven of eight models, "
        "and the best identification in seven of eight.", 34)),
        fontsize=22, color=INK, va="top", linespacing=1.45)
    ax.text(0.62, 0.44, "\n".join(textwrap.wrap(
        "Late-layer injection is barely above chance throughout, a concept inserted "
        "close to the output has too little depth left to be integrated into anything "
        "reportable.", 36)),
        fontsize=19, color=MUTED, va="top", linespacing=1.45)
    return save(fig, "slide_09.png")


def slide_10():
    fig, ax = new_slide()
    heading(ax, "Open source, and what's next")
    rule(ax)
    ax.text(0.07, 0.755, "Self-report about the model's own situation is the least\n"
                         "reliable measurement in every part of this study.",
            fontsize=28, color=ACCENT, weight="bold", va="top", linespacing=1.5)
    ax.text(0.07, 0.585, "It is the measurement AI-welfare claims most depend on.",
            fontsize=23, color=INK, va="top")
    bullets(ax, [
        "Every number regenerates from committed data, no GPU. Two scripts reproduce "
        "every experiment, and a smoke test exercises every code path in about a minute.",
        "Next: an established outcome set, injection beyond 14 billion parameters and "
        "more concepts, and whether better self-prediction goes with better detection.",
    ], y0=0.47, dy=0.17, size=21, wrap=84)
    ax.text(0.07, 0.135, "github.com/arpitsinghgautam/selfprobe",
            fontsize=19, color=MUTED, style="italic")
    ax.text(0.07, 0.065, "Cross-model studies of self-knowledge need a within-model contrast.",
            fontsize=24, color=ACCENT, weight="bold")
    return save(fig, "slide_10.png")


# --------------------------------------------------------------------------- #

def main() -> None:
    builders = [slide_01, slide_02, slide_03, slide_04, slide_05,
                slide_06, slide_07, slide_08, slide_09, slide_10]
    paths = []
    for b in builders:
        p = b()
        paths.append(p)
        print(f"  {p.relative_to(ROOT)}")

    # Slideshow PDF from the same frames, no extra dependency needed.
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in paths]
    REPORT.mkdir(exist_ok=True)
    pdf = REPORT / "deck_p2.pdf"
    imgs[0].save(str(pdf), save_all=True, append_images=imgs[1:])
    print(f"\n  {pdf.relative_to(ROOT)}  ({pdf.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
