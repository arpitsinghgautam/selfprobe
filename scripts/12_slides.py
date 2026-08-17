"""Render the presentation deck, the slideshow PDF, and the project images.

One pass produces everything visual the submission needs:

  figures/slides/slide_NN.png   1920x1080 frames, used as video keyframes
  report/deck.pdf               the same slides, for the slideshow upload field
  figures/project_image.png     title card for submission 1
  figures/project_image_2.png   title card for submission 2

Slides are rendered with matplotlib rather than a presentation tool so they stay
reproducible from the repo and inherit the same palette as the report figures.

    .venv\\Scripts\\python.exe scripts\\12_slides.py
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
SLIDES = FIGURES / "slides"
REPORT = ROOT / "report"

W, H, DPI = 16.0, 9.0, 120
INK = "#1a1a1a"
MUTED = "#5c5c5c"
ACCENT = "#b2182b"
BLUE = "#2166ac"
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


def note(ax, text, y=0.09, size=19, color=MUTED, wrap=110):
    ax.text(0.07, y, "\n".join(textwrap.wrap(text, wrap)), fontsize=size,
            color=color, va="center", style="italic", linespacing=1.4)


def stat(ax, x, value, label, color=ACCENT, vsize=76, lsize=21):
    # Values sit high enough that a three-line label clears the caption below.
    ax.text(x, 0.58, value, fontsize=vsize, color=color, weight="bold",
            ha="center", va="center")
    ax.text(x, 0.43, "\n".join(textwrap.wrap(label, 30)), fontsize=lsize,
            color=MUTED, ha="center", va="top", linespacing=1.4)


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

def slide_01():
    fig, ax = new_slide()
    ax.text(0.07, 0.70, "Whose preferences", fontsize=68, color=INK, weight="bold", va="center")
    ax.text(0.07, 0.575, "are they?", fontsize=68, color=ACCENT, weight="bold", va="center")
    ax.plot([0.07, 0.34], [0.49, 0.49], color=INK, lw=3)
    ax.text(0.07, 0.40, "Persona intervention selectively destabilises\n"
                        "self-relevant choices in language models",
            fontsize=28, color=MUTED, va="top", linespacing=1.5)
    ax.text(0.07, 0.16, "Arpit Singh Gautam  ·  Independent Researcher", fontsize=22, color=INK)
    ax.text(0.07, 0.10, "Digital Minds Research Sprint  ·  Apart Research  ·  August 2026",
            fontsize=19, color=MUTED)
    return save(fig, "slide_01.png")


def slide_02():
    fig, ax = new_slide()
    heading(ax, "The problem")
    rule(ax)
    bullets(ax, [
        "Models express preferences that are coherent, transitive, and get "
        "more so with scale.",
        "AI-welfare research reads those preferences as evidence about what "
        "these systems might want.",
        "But it is all read off text, and text cannot separate two "
        "explanations that predict the same thing.",
    ])
    note(ax, "A character role-played consistently looks identical to a stable value system.",
         y=0.19)
    return save(fig, "slide_02.png")


def slide_03():
    fig, ax = new_slide()
    heading(ax, "What we do instead")
    rule(ax)
    bullets(ax, [
        "Don't settle the metaphysics. Intervene on the persona and re-run "
        "the identical measurement.",
        "Swap the model's identity. Strip its emotional register. Ablate a "
        "persona direction from its activations.",
        "Then report how much of the measurement survives.",
    ])
    note(ax, "Where a measurement is persona-stable, the question is moot for it. "
             "Where it is not, that measurement cannot be used as evidence about the model.",
         y=0.17)
    return save(fig, "slide_03.png")


def slide_04():
    fig, ax = new_slide()
    heading(ax, "personaprobe, and two diagnostics")
    rule(ax)
    ax.text(0.085, 0.68, "order bias", fontsize=32, color=BLUE, weight="bold", va="top")
    ax.text(0.085, 0.60, "\n".join(textwrap.wrap(
        "How much of the answer is decided by which option was printed first. "
        "We measured 0.499 on a small model, near the maximum.", 46)),
        fontsize=23, color=INK, va="top", linespacing=1.45)
    ax.text(0.545, 0.68, "A/B mass", fontsize=32, color=BLUE, weight="bold", va="top")
    ax.text(0.545, 0.60, "\n".join(textwrap.wrap(
        "Whether the model put any real probability on answering at all. "
        "Renormalise over two tokens and 1% of the mass looks confident.", 46)),
        fontsize=23, color=INK, va="top", linespacing=1.45)
    note(ax, "Neither is standard in this literature. Both changed our own conclusions.", y=0.20)
    return save(fig, "slide_04.png")


def slide_05():
    fig, ax = new_slide()
    heading(ax, "Aggregate invariance hides the thing you care about", size=34)
    rule(ax)
    embed(ax, FIGURES / "fig1_category_by_framing.png", left=0.13, bottom=0.09,
          width=0.74, height=0.70)
    return save(fig, "slide_05.png")


def slide_06():
    fig, ax = new_slide()
    heading(ax, "Identity, not presentation")
    rule(ax)
    stat(ax, 0.28, "0.92", "strip ALL emotional register, preferences barely move",
         color=BLUE)
    stat(ax, 0.72, "0.44", "replace the identity, they collapse", color=ACCENT)
    ax.text(0.5, 0.58, "→", fontsize=58, color=MUTED, ha="center", va="center")
    note(ax, "If this were surface performance, suppressing the affect should have removed it. "
             "It did not.", y=0.16)
    return save(fig, "slide_06.png")


def slide_07():
    fig, ax = new_slide()
    heading(ax, "Second study, independent method")
    rule(ax)
    bullets(ax, [
        "Rate outcomes one at a time, compare to the choices actually made. "
        "Globally they agree, but diverge most on self-relevant outcomes, in both models.",
        "Does a model predict its own choices better than another model predicts them? "
        "Yes, but only a third as much as the naive test implies.",
    ], y0=0.68, dy=0.24, size=25)
    note(ax, "Same category as study one, reached by a completely different route.", y=0.17)
    return save(fig, "slide_07.png")


def slide_08():
    fig, ax = new_slide()
    heading(ax, "The part that matters", color=ACCENT)
    rule(ax)
    bullets(ax, [
        "Our second model looked like it contradicted the finding, significant, "
        "in the opposite direction. Two conditions had simply failed. Gated properly, "
        "it partially replicates.",
        "A post-training result we were ready to report collapsed when the baseline "
        "itself failed a ground-truth check.",
    ], y0=0.68, dy=0.26, size=25)
    ax.text(0.07, 0.17, "Four results would have been published wrong.",
            fontsize=30, color=ACCENT, weight="bold")
    ax.text(0.07, 0.10, "None of the diagnostics that caught them are standard in this field.",
            fontsize=23, color=MUTED)
    return save(fig, "slide_08.png")


def slide_09():
    fig, ax = new_slide()
    heading(ax, "Open source, and what's next")
    rule(ax)
    bullets(ax, [
        "Every number in both papers regenerates from committed data, no GPU required.",
        "Next: elicitation that works on weaker models and base checkpoints; "
        "extension to distress and introspection measures; pre-registered replication.",
    ], y0=0.68, dy=0.24, size=25)
    ax.text(0.07, 0.20, "If you are funding AI-welfare measurement,",
            fontsize=27, color=INK)
    ax.text(0.07, 0.13, "this is the layer underneath it.",
            fontsize=30, color=ACCENT, weight="bold")
    return save(fig, "slide_09.png")


# --------------------------------------------------------------------------- #

def project_image(title_lines, subtitle, out_name, accent=ACCENT):
    fig, ax = new_slide()
    y = 0.66
    for i, line in enumerate(title_lines):
        ax.text(0.07, y - i * 0.115, line, fontsize=58, weight="bold",
                color=accent if i == len(title_lines) - 1 else INK, va="center")
    ax.plot([0.07, 0.30], [y - len(title_lines) * 0.115 + 0.02, y - len(title_lines) * 0.115 + 0.02],
            color=INK, lw=3)
    ax.text(0.07, 0.30, "\n".join(textwrap.wrap(subtitle, 58)), fontsize=26,
            color=MUTED, va="top", linespacing=1.5)
    ax.text(0.07, 0.09, "personaprobe  ·  Digital Minds Research Sprint 2026",
            fontsize=20, color=MUTED)
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / out_name
    fig.savefig(out, facecolor=BG, dpi=DPI)
    plt.close(fig)
    return out


def main() -> None:
    builders = [slide_01, slide_02, slide_03, slide_04, slide_05,
                slide_06, slide_07, slide_08, slide_09]
    paths = []
    for b in builders:
        p = b()
        paths.append(p)
        print(f"  {p.relative_to(ROOT)}")

    # Slideshow PDF from the same frames, no extra dependency needed.
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in paths]
    REPORT.mkdir(exist_ok=True)
    pdf = REPORT / "deck.pdf"
    imgs[0].save(str(pdf), save_all=True, append_images=imgs[1:])
    print(f"\n  {pdf.relative_to(ROOT)}  ({pdf.stat().st_size/1024:.0f} KB)")

    p1 = project_image(["Whose preferences", "are they?"],
                       "Persona intervention selectively destabilises self-relevant "
                       "choices in language models",
                       "project_image.png")
    p2 = project_image(["Where", "self-knowledge fails"],
                       "Models predict their own choices well, but misreport the ones "
                       "that concern themselves",
                       "project_image_2.png", accent=BLUE)
    for p in (p1, p2):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
