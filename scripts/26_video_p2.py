"""Assemble the project-2 presentation video: slides + narration -> MP4.

Same machinery as scripts/13_video.py — edge-tts for the read, ffmpeg to hold
each slide for exactly as long as its narration — pointed at the selfprobe deck
so the two submissions can be rebuilt independently of each other.

    .venv\\Scripts\\python.exe scripts\\26_video_p2.py

Re-recording in your own voice later does not require redoing any of this: play
report/deck_p2.pdf fullscreen, narrate over it with Game Bar (Win+G), and the
script below is the read.

Every number spoken here comes from report/report2.md. Digits are spelled out
because the TTS voice otherwise reads "0.548" as "point five four eight" or
worse; letters are hyphenated ("A-I", "G-P-U") for the same reason.

Outputs video/p2/selfprobe.mp4 plus per-slide MP3s under video/p2/audio/.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLIDES = ROOT / "figures" / "slides_p2"
VIDEO = ROOT / "video" / "p2"
AUDIO = VIDEO / "audio"

# Same voice as the other deck — the two videos get watched back to back.
VOICE_PREFERENCES = [
    "en-US-AndrewNeural",
    "en-GB-RyanNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
]

# One entry per slide, in order. Keep these in sync with scripts/25_slides_p2.py.
NARRATION: list[str] = [
    # 1 — title
    "Claims about A-I welfare lean heavily on what a model reports about itself — that a shutdown "
    "is bad, that a task is distressing. We put that reporting to the test three structurally "
    "different ways, then placed a known concept into the activations and asked whether the model "
    "noticed.",

    # 2 — the two assumptions
    "The reading rests on two separable assumptions. First, that a model's stated evaluation "
    "matches how it would actually behave — it can rate an outcome as terrible and still choose "
    "it. Second, that the model is a better source about itself than an outsider is: if another "
    "model predicts its choices equally well, the self-report carries nothing privileged. Neither "
    "is usually tested.",

    # 3 — the design
    "So we ask for the same preferences three ways over identical material: forty outcomes, eight "
    "of them about the model itself, plus a donation ladder whose correct ordering is known "
    "without consulting any model. Revealed is forced pairwise choice over seven hundred and "
    "eighty pairs. Stated is a one-at-a-time rating on a five-point letter scale. Predicted asks "
    "which option a described chooser will take — impersonal, so the same question can go to "
    "another model.",

    # 4 — stated vs revealed
    "In aggregate the elicitations agree: zero point eight seven two for the seven-billion Qwen, "
    "zero point eight two eight for Mistral. Both reproduce the donation ladder perfectly, so the "
    "rating instrument works. Split by category and the disagreement has a location: self-relevant "
    "outcomes are the worst-agreeing substantive category in both models, at zero point six four "
    "three and zero point five four eight. A companion submission finds the same category least "
    "stable under persona intervention, by an unrelated method.",

    # 5 — privileged access and the noise confound
    "Privileged access. The standard test asks whether a model predicts its own choices better "
    "than another model does. Ours scores zero point nine four eight against zero point eight six "
    "two. But the external predictor is a worse instrument — order bias zero point five six one, "
    "answer mass zero point five three eight — and a noisier predictor loses at predicting "
    "anything, including itself. So hold quality fixed: the same model predicts an A-I assistant, "
    "then a different A-I assistant. Zero point nine four eight versus zero point nine one seven. "
    "What survives is zero point zero three one — a third of what the naive test implies.",

    # 6 — the injection benchmark
    "Now the measurement with ground truth. Six concepts, each a direction built from four "
    "sentences that evoke it minus four matched neutral ones. We add it to the residual stream at "
    "three depths and six strengths — one of which is zero. That cell is the false-alarm baseline: "
    "the identical question with nothing injected. Every cell also carries its answer mass, "
    "because an affirmative answer from a model that has stopped answering is not introspection.",

    # 7 — the false-alarm result
    "This is the one to take away. Falcon three seven B reports an injected concept fifty six "
    "point one percent of the time — and fifty four point seven percent of the time when nothing "
    "is injected. Discrimination: zero point zero one four. The half-billion Qwen is worse, at "
    "sixty one point two and sixty two point two, giving minus zero point zero one one. Those are "
    "the two highest raw detection rates we measured. Report true positives alone and you rank "
    "them first; the false-alarm baseline ranks them last.",

    # 8 — detection vs identification
    "Detection and identification also come apart. The fourteen-billion Qwen has a discrimination "
    "of zero point zero four nine — it almost never reports noticing anything. Yet it has the "
    "highest identification accuracy in the set: zero point six zero nine against chance of zero "
    "point five. The injected concept shapes its forced choices while its own report says nothing "
    "is there.",

    # 9 — depth
    "Depth matters, and behaves consistently across families. Middle-depth injection gives the "
    "best detection in seven of eight models, and the best identification in seven of eight. "
    "Late-layer injection is barely above chance throughout — too little depth left, that close to "
    "the output, to integrate the concept into anything reportable.",

    # 10 — close
    "Every effect here is small, and I want to be plain about that. Identification tops out at "
    "zero point six zero nine; the advantage that survives the noise control is zero point zero "
    "three one. Measurable, not substantial. But the direction is consistent: self-report about "
    "the model's own situation is the least reliable measurement here, and the one A-I welfare "
    "claims most depend on. Everything is open source, and every number regenerates without a "
    "G-P-U. Cross-model studies of self-knowledge need a within-model contrast as standard. "
    "Thank you.",
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
        raise SystemExit(f"{len(slides)} slides but {len(audios)} narration clips — "
                         "keep 25_slides_p2.py and NARRATION in sync")

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

    out = VIDEO / "selfprobe.mp4"
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
    VIDEO.mkdir(parents=True, exist_ok=True)
    ffmpeg, ffprobe = ffmpeg_bin("ffmpeg"), ffmpeg_bin("ffprobe")

    voice = asyncio.run(pick_voice())
    print(f"voice: {voice}\n")
    audios = asyncio.run(synthesize(voice))
    print()
    build(ffmpeg, ffprobe, audios)


if __name__ == "__main__":
    main()
