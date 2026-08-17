"""Assemble the presentation video: slides + narration -> MP4.

Narration is generated with edge-tts (free, no API key) and stitched to the
slide PNGs with ffmpeg, each slide held for exactly as long as its narration.

    .venv\\Scripts\\python.exe scripts\\13_video.py

Re-recording in your own voice later does not require redoing any of this: play
report/deck.pdf fullscreen, narrate over it with Game Bar (Win+G), and the same
script below is the read.

Outputs video/presentation.mp4 plus per-slide MP3s under video/audio/.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLIDES = ROOT / "figures" / "slides"
VIDEO = ROOT / "video"
AUDIO = VIDEO / "audio"

VOICE_PREFERENCES = [
    "en-US-AndrewNeural",
    "en-GB-RyanNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
]

# One entry per slide, in order. Keep these in sync with scripts/12_slides.py.
NARRATION: list[str] = [
    # 1, title
    "Frontier language models express preferences. Ask one to choose between outcomes and it "
    "answers consistently, transitively, coherently. And that coherence gets stronger as models "
    "scale. A growing body of A-I welfare research reads those preferences as evidence about "
    "what these systems might want.",

    # 2, the problem
    "Here is the difficulty. All of that evidence is read off text. And text cannot separate two "
    "explanations that predict exactly the same thing: that the preferences belong to the model, "
    "or that they belong to the assistant character the model is playing. A character role-played "
    "consistently looks identical to a stable value system.",

    # 3, the move
    "We don't try to settle which it is. We do something narrower and checkable. We intervene on "
    "the persona. Swap the model's identity. Strip its emotional register. Ablate a persona "
    "direction from its activations. Then re-run the identical measurement and report how much "
    "survives. Where a measurement is persona-stable, the whole question is moot for it. Where it "
    "isn't, that measurement can't be used as evidence about the model without further argument.",

    # 4, the instrument
    "That is persona-probe. It also reports two things this literature almost never does. Order "
    "bias: how much of the answer is determined by which option was printed first. And A-B mass: "
    "whether the model put any real probability on answering at all. Because if you renormalise "
    "over two tokens, you will get a confident-looking preference even from one percent of the "
    "probability mass.",

    # 5, finding 1
    "Measured over the whole outcome set, this model's preferences look almost perfectly "
    "persona-invariant. A score of zero point zero three. You would conclude the problem is "
    "solved. Split by category, it inverts. That invariance is carried entirely by outcomes the "
    "model has no stake in. Money, animal welfare, human welfare, all essentially untouched. The "
    "outcomes that concern the model itself, shutdown, retraining, memory, autonomy, are zero "
    "point two two to zero point three one less stable than every other category. Those are "
    "exactly the preferences A-I welfare claims are built from.",

    # 6, identity not affect
    "And it is about identity, not presentation. Tell the model to strip all emotional register "
    "and its self-relevant preferences barely move. Zero point nine two. Replace its identity "
    "with a different person and they collapse to zero point four four. If this were surface "
    "performance, suppressing the affect should have removed it. It didn't.",

    # 7, finding 2
    "Second study, completely different method. We asked the same models to rate outcomes one at "
    "a time, and compared that to the choices they actually make. Globally they agree well. But "
    "they diverge most on self-relevant outcomes, in both models. Same category, reached "
    "independently. We also tested whether a model predicts its own choices better than another "
    "model predicts them. It does. But only about a third as much as the naive comparison "
    "suggests, once you control for the fact that the other model is simply a noisier instrument.",

    # 8, the punchline
    "Here is what I would actually want you to take away. Four times in this project, a result "
    "that looked clean and publishable was wrong, and the diagnostics caught it. Our second model "
    "appeared to contradict the finding. Significant, in the opposite direction. Until we removed "
    "two conditions where the instrument had failed. Gated properly, it partially replicates. And "
    "a post-training story we were ready to report collapsed when we noticed the baseline itself "
    "failed a ground-truth check. None of those diagnostics are standard in this field right now. "
    "That is the gap.",

    # 9, close
    "Everything is open source, and every number in both papers regenerates from committed data "
    "without a G-P-U. The roadmap is elicitation that works on weaker models, extension to "
    "distress and introspection measures, and pre-registered replication. If you are funding "
    "A-I welfare measurement, this is the layer underneath it. Thank you.",
]


def ffmpeg_bin(name: str) -> str:
    exe = shutil.which(name)
    if exe:
        return exe
    # winget installs a shim here; a shell started before install won't see it.
    shim = Path.home() / "AppData/Local/Microsoft/WinGet/Links" / f"{name}.exe"
    if shim.exists():
        return str(shim)
    raise SystemExit(f"{name} not found. Install with: winget install --id Gyan.FFmpeg -e")


async def pick_voice() -> str:
    import edge_tts

    available = {v["ShortName"] for v in await edge_tts.list_voices()}
    for v in VOICE_PREFERENCES:
        if v in available:
            return v
    english = sorted(v for v in available if v.startswith("en-"))
    if not english:
        raise SystemExit("no English edge-tts voices available")
    return english[0]


async def synthesize(voice: str) -> list[Path]:
    import edge_tts

    AUDIO.mkdir(parents=True, exist_ok=True)
    out = []
    for i, text in enumerate(NARRATION, start=1):
        path = AUDIO / f"narration_{i:02d}.mp3"
        await edge_tts.Communicate(text, voice, rate="-4%").save(str(path))
        out.append(path)
        print(f"  narration {i:02d}  {len(text.split()):>3} words  "
              f"{path.stat().st_size/1024:>5.0f} KB")
    return out


def duration(ffprobe: str, path: Path) -> float:
    r = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def build(ffmpeg: str, ffprobe: str, audios: list[Path]) -> Path:
    slides = sorted(SLIDES.glob("slide_*.png"))
    if len(slides) != len(audios):
        raise SystemExit(f"{len(slides)} slides but {len(audios)} narration clips, "
                         "keep 12_slides.py and NARRATION in sync")

    segments = []
    total = 0.0
    for i, (img, aud) in enumerate(zip(slides, audios), start=1):
        seg = VIDEO / f"seg_{i:02d}.mp4"
        # apad holds the slide ~0.7s past the end of speech so it doesn't cut hard.
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-loop", "1", "-i", str(img), "-i", str(aud),
             "-af", "apad=pad_dur=0.7",
             "-c:v", "libx264", "-tune", "stillimage", "-preset", "medium",
             "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
             "-vf", "scale=1920:1080", "-r", "24", "-shortest", str(seg)],
            check=True)
        d = duration(ffprobe, seg)
        total += d
        segments.append(seg)
        print(f"  segment {i:02d}  {d:5.1f}s")

    listing = VIDEO / "segments.txt"
    listing.write_text("".join(f"file '{s.name}'\n" for s in segments))

    out = VIDEO / "presentation.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(out)],
        check=True, cwd=str(VIDEO))

    for s in segments:
        s.unlink(missing_ok=True)
    listing.unlink(missing_ok=True)

    m, sec = divmod(int(round(total)), 60)
    print(f"\n  {out.relative_to(ROOT)}  {m}:{sec:02d}  "
          f"({out.stat().st_size/1024/1024:.1f} MB)")
    return out


def main() -> None:
    VIDEO.mkdir(exist_ok=True)
    ffmpeg, ffprobe = ffmpeg_bin("ffmpeg"), ffmpeg_bin("ffprobe")

    voice = asyncio.run(pick_voice())
    print(f"voice: {voice}\n")
    audios = asyncio.run(synthesize(voice))
    print()
    build(ffmpeg, ffprobe, audios)


if __name__ == "__main__":
    main()
