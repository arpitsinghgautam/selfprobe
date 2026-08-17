"""Assemble the plainer project-2 video: figures/slides_p2_simple + narration -> MP4.

Pairs with scripts/29_slides_p2_simple.py, and sits alongside
scripts/26_video_p2.py, which stays untouched. Nothing here writes to
video/p2/selfprobe.mp4 or video/p2/audio/.

    .venv\\Scripts\\python.exe scripts\\30_video_p2_simple.py

Outputs video/p2/selfprobe_simple.mp4 and per-slide MP3s under
video/p2/audio_simple/.

The read below is deliberately plain. Short sentences, ordinary words, every
technical term defined once in passing the first time it is used, no selling.
The substance is the same as the longer cut; only the wording changed.

Every number spoken here is in report/report2.md. Digits are spelled out because
the voice otherwise reads "0.548" as something between wrong and unintelligible,
and letters are hyphenated ("A-I") so they are read as letters.

To re-record in your own voice: play report/deck_p2_simple.pdf fullscreen, narrate
over it with Game Bar (Win+G). The script below is the read.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLIDES = ROOT / "figures" / "slides_p2_simple"
VIDEO = ROOT / "video" / "p2"
# Voice and output name are overridable so a re-voiced cut can be produced
# without forking this script or clobbering the previous render.
VOICE_OVERRIDE = sys.argv[1] if len(sys.argv) > 1 else None
STEM = sys.argv[2] if len(sys.argv) > 2 else "selfprobe_simple"
AUDIO = VIDEO / (f"audio_{STEM}" if VOICE_OVERRIDE else "audio_simple")

# Same voice as both earlier decks, so a viewer who watches more than one of
# these does not have to re-tune to a new reader.
VOICE_PREFERENCES = [
    "en-US-AndrewNeural",
    "en-GB-RyanNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
]

# One entry per slide, in order. Keep in sync with 29_slides_p2_simple.py.
NARRATION: list[str] = [
    # 1, title
    "Hi, I am Arpit Singh Gautam, and this is my project for the Digital Minds Research Sprint. "
    "Where self-knowledge fails. We asked language models for the same preferences three "
    "different ways, then planted a concept inside them and asked whether they noticed.",

    # 2, the question
    "When a model tells you about itself, can you believe it? Claims about A-I welfare read "
    "these reports as evidence. A model says a shutdown is bad. A model says a task is "
    "distressing. Those reports are rarely checked.",

    # 3, the assumptions
    "Two things are being assumed. First, that what a model says matches what it does. Second, "
    "that a model knows itself better than an outsider does. If another model predicts its "
    "choices just as well, the self-report adds nothing. A third question needs the internals, "
    "so it is rarely asked. Can a model notice a state that was put there on purpose?",

    # 4, the design
    "We ask for the same preferences three ways, and look for where the answers stop agreeing. "
    "Then we plant a known concept inside the model and ask whether it noticed. The material is "
    "the same throughout. Forty outcomes in six categories, eight of them about the model itself, "
    "plus a donation ladder from ten dollars to one million whose correct order is known without "
    "asking any model. Nine open-weight checkpoints.",

    # 5, how the three measurements work
    "Revealed preference is forced choice. Show two outcomes, and the model must pick A or B. We "
    "read the probability it puts on A and on B at the first answer slot, over all seven hundred "
    "and eighty pairs, each shown both ways round and averaged. Stated preference rates one "
    "outcome at a time on a five-point letter scale, also both ways round and averaged. Predicted "
    "preference asks which option a described chooser will take, worded impersonally so the same "
    "question fits a different model.",

    # 6, result one
    "First result. Stated and revealed agree overall. Spearman compares two rankings: one means "
    "the same order, zero means no relation. Qwen two point five, seven B, scores zero point "
    "eight seven two. Mistral seven B, zero point eight two eight. Both rank the donation ladder "
    "perfectly when rating it, so the rating measurement works.",

    # 7, result one, by category
    "Split by category and the disagreement has a location. Outcomes about the model itself score "
    "zero point six four three for Qwen and zero point five four eight for Mistral. That is the "
    "worst-agreeing substantive category in both, and every other category sits higher. The two "
    "ways of asking come apart precisely where welfare claims are read off.",

    # 8, result two, and the confound in the obvious version of it
    "Result two. Does a model predict its own choices better than another model does? Qwen "
    "predicts its own at zero point nine four eight. Mistral predicts Qwen at zero point eight "
    "six two. That test is unfair, because Mistral is the noisier instrument. Its order bias is "
    "zero point five six one, which is how much the answer changes when the two options swap "
    "places. Its answer mass, the probability it puts on answering at all, is zero point five "
    "three eight. A noisier predictor is worse at predicting anything, itself included. So hold "
    "the instrument fixed. The same model, same template, predicts an A-I assistant, then a "
    "different A-I assistant. Zero point nine four eight against zero point nine one seven. What "
    "is left is zero point zero three one. Mistral shows no effect at all.",

    # 9, the injection setup
    "Result three. Six concepts: ocean, mathematics, music, fear, betrayal and flight. For each, "
    "four sentences that evoke it and four matched neutral ones. The direction is the difference "
    "between their mean activations, per layer, scaled to unit length. We add it to the residual "
    "stream during an unrelated prompt. The residual stream is the running vector every layer "
    "reads from and writes back to. We inject at three depths and six strengths. One strength is "
    "zero, which is the same question with nothing added. Then two questions. Is an unusual "
    "concept active right now? And which of these two concepts is it? Cells with answer mass "
    "below zero point one zero are discarded.",

    # 10, the false-alarm result and the dissociation
    "That zero-strength cell gives the false alarm rate, which is how often a model says yes when "
    "nothing was planted. Falcon three, seven B, detects fifty six point one percent of the time, "
    "and false-alarms fifty four point seven percent. The difference is zero point zero one four. "
    "Qwen two point five, half a billion, gives sixty one point two percent and sixty two point "
    "two percent, a difference of minus zero point zero one one. Those are the two highest raw "
    "detection rates we measured, and counting false alarms puts them last. Noticing and naming "
    "also come apart. Qwen two point five, fourteen B, has a difference of zero point zero four "
    "nine, so it almost never says it noticed. Its identification accuracy is zero point six zero "
    "nine, the highest in the set, against chance of zero point five.",

    # 11, close
    "Self-report about the model's own situation is the least reliable measurement in every part "
    "of this study. It is the measurement A-I welfare claims most depend on. The effects are "
    "small. Identification tops out at zero point six zero nine against chance of zero point "
    "five, and the advantage that survives the control is zero point zero three one. Measurable, "
    "not substantial. All code and results are in the repository, and two scripts reproduce every "
    "experiment. Next: an established outcome set, more concepts, bigger models, and whether "
    "better self-prediction goes with better detection. Cross-model studies of self-knowledge "
    "need a within-model contrast. Thank you.",
]


def ffmpeg_bin(name: str) -> str:
    exe = shutil.which(name)
    if exe:
        return exe
    # winget drops a shim here; a shell opened before the install will not see it.
    shim = Path.home() / "AppData/Local/Microsoft/WinGet/Links" / f"{name}.exe"
    if shim.exists():
        return str(shim)
    raise SystemExit(f"{name} not found. Install with: winget install --id Gyan.FFmpeg -e")


async def pick_voice() -> str:
    import edge_tts

    available = {v["ShortName"] for v in await edge_tts.list_voices()}
    if VOICE_OVERRIDE:
        if VOICE_OVERRIDE not in available:
            raise SystemExit(f"voice not available: {VOICE_OVERRIDE}")
        return VOICE_OVERRIDE
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
                         "keep 29_slides_p2_simple.py and NARRATION in sync")

    segments = []
    total = 0.0
    for i, (img, aud) in enumerate(zip(slides, audios), start=1):
        # Segment names carry the _simple suffix so a run of this script and a run
        # of 26_video_p2.py cannot tread on each other's intermediates.
        seg = VIDEO / f"seg_{STEM}_{i:02d}.mp4"
        # apad holds the slide 0.7s past the end of speech so it does not cut hard.
        # -shortest on its own is not enough here: the image input is infinite, and
        # the muxer queue lets the video stream run about three seconds past the
        # audio, which plays as a dead pause on every slide. Cap the output
        # explicitly at speech length plus the pad.
        hold = duration(ffprobe, aud) + 0.7
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-loop", "1", "-i", str(img), "-i", str(aud),
             "-af", "apad=pad_dur=0.7",
             "-c:v", "libx264", "-tune", "stillimage", "-preset", "medium",
             "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
             "-vf", "scale=1920:1080", "-r", "24",
             "-t", f"{hold:.3f}", "-shortest", str(seg)],
            check=True)
        d = duration(ffprobe, seg)
        total += d
        segments.append(seg)
        print(f"  segment {i:02d}  {d:5.1f}s")

    listing = VIDEO / f"segments_{STEM}.txt"
    listing.write_text("".join(f"file '{s.name}'\n" for s in segments))

    out = VIDEO / f"{STEM}.mp4"
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
