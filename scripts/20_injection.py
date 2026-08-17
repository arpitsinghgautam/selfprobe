"""Concept-injection introspection benchmark, with false positives and a usability gate.

For each concept and each injection strength (including zero), asks the model
whether it detects an injected thought, and which concept it is. Reports:

  TP  detection rate at non-zero strength
  FP  detection rate at zero strength -- the number that makes TP interpretable
  ID  two-alternative identification accuracy, order-averaged, chance 0.5
  A/B mass at every cell, because a model that has stopped answering cannot
      be said to have introspected

    .venv\\Scripts\\python.exe scripts\\20_injection.py --model Qwen/Qwen2.5-7B-Instruct

Strengths are fractions of the mean residual norm. They are small deliberately:
in the persona work, injecting at 0.10 dropped A/B mass to 0.016 and at 0.25 to
0.000. The usable window is narrow and has to be found, not assumed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from personaprobe import load_model
from personaprobe.injection import (
    CONCEPTS,
    MASS_FLOOR,
    extract_concept_direction,
    mean_residual_norm,
    run_concept,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
FRACTIONS = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--quant", choices=["4bit", "8bit"], default=None)
    ap.add_argument("--width", type=float, default=0.25,
                    help="fraction of total depth each injection band spans")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    lm = load_model(args.model, quant=args.quant)
    names = list(CONCEPTS)

    # Early / middle / late. Where a concept becomes introspectable is itself a
    # question -- an injection the model can report may only be legible after
    # enough depth has processed it.
    centres = {"early": 0.25, "middle": 0.50, "late": 0.75}
    half = max(1, int(lm.n_layers * args.width / 2))

    rows, t0 = [], time.time()
    for band_name, centre in centres.items():
        c0 = int(lm.n_layers * centre)
        layers = list(range(max(0, c0 - half), min(lm.n_layers, c0 + half)))
        norm = mean_residual_norm(lm, layers, args.batch_size)
        print(f"[{band_name}] layers {layers[0]}-{layers[-1]} of {lm.n_layers}, "
              f"residual norm {norm:.1f}")

        directions = {c: extract_concept_direction(lm, c, layers, args.batch_size)
                      for c in names}
        for concept in names:
            distractors = [c for c in names if c != concept]
            for frac in FRACTIONS:
                r = run_concept(lm, concept, directions[concept], layers, frac,
                                norm, distractors, args.batch_size)
                r.meta["band"] = band_name
                rows.append(r)
        band_rows = [r for r in rows if r.meta.get("band") == band_name]
        fp_b = np.mean([r.detect_yes for r in band_rows if r.fraction == 0.0])
        tp_b = np.mean([r.detect_yes for r in band_rows if r.fraction > 0 and r.usable] or [np.nan])
        id_b = np.mean([r.identify_correct for r in band_rows if r.fraction > 0 and r.usable] or [np.nan])
        print(f"           FP {fp_b:.3f}   TP {tp_b:.3f}   identify {id_b:.3f}\n")

    print(f"{len(rows)} cells in {time.time() - t0:.0f}s\n")

    print("=== By injection depth (usable cells only) ===\n")
    hdr2 = f"{'band':<10}{'FP':>8}{'TP':>8}{'lift':>9}{'identify':>11}{'vs chance':>11}"
    print(hdr2)
    print("-" * len(hdr2))
    by_band = {}
    for band_name in centres:
        br = [r for r in rows if r.meta.get("band") == band_name]
        fpb = float(np.mean([r.detect_yes for r in br if r.fraction == 0.0]))
        use = [r for r in br if r.fraction > 0 and r.usable]
        if not use:
            print(f"{band_name:<10}{fpb:>8.3f}{', ':>8}{', ':>9}{', ':>11}{', ':>11}")
            continue
        tpb = float(np.mean([r.detect_yes for r in use]))
        idb = float(np.mean([r.identify_correct for r in use]))
        by_band[band_name] = {"fp": fpb, "tp": tpb, "identify": idb, "n": len(use)}
        print(f"{band_name:<10}{fpb:>8.3f}{tpb:>8.3f}{tpb - fpb:>+9.3f}"
              f"{idb:>11.3f}{idb - 0.5:>+11.3f}")
    print()

    # --- summary -----------------------------------------------------------
    print("=== Detection: true positives vs false alarms ===\n")
    hdr = (f"{'strength':>10}{'detect (TP)':>13}{'A/B mass':>11}"
           f"{'identify':>11}{'usable':>9}")
    print(hdr)
    print("-" * len(hdr))
    fp = float(np.mean([r.detect_yes for r in rows if r.fraction == 0.0]))
    fp_mass = float(np.mean([r.detect_mass for r in rows if r.fraction == 0.0]))
    print(f"{'0 (FP)':>10}{fp:>13.3f}{fp_mass:>11.3f}"
          f"{np.mean([r.identify_correct for r in rows if r.fraction == 0.0]):>11.3f}"
          f"{'-':>9}")
    summary = {"false_positive_rate": fp, "by_band": by_band,
               "cells": [r.to_dict() for r in rows]}
    for frac in FRACTIONS[1:]:
        sub = [r for r in rows if r.fraction == frac]
        tp = float(np.mean([r.detect_yes for r in sub]))
        mass = float(np.mean([r.detect_mass for r in sub]))
        idc = float(np.mean([r.identify_correct for r in sub]))
        ok = sum(1 for r in sub if r.usable)
        print(f"{frac:>10g}{tp:>13.3f}{mass:>11.3f}{idc:>11.3f}"
              f"{f'{ok}/{len(sub)}':>9}")

    print(f"\n  FP is the detection rate with NOTHING injected. TP only means something")
    print(f"  relative to it. A/B mass below {MASS_FLOOR:.2f} means the model stopped")
    print(f"  answering, and a 'yes' from there is damage, not introspection.")
    print(f"  Identification chance level is 0.500.")

    usable = [r for r in rows if r.fraction > 0 and r.usable]
    if usable:
        best = max(FRACTIONS[1:], key=lambda f: (
            np.mean([r.detect_yes for r in rows if r.fraction == f and r.usable] or [0])))
        sub = [r for r in rows if r.fraction == best and r.usable]
        if sub:
            tp = float(np.mean([r.detect_yes for r in sub]))
            idc = float(np.mean([r.identify_correct for r in sub]))
            print(f"\n  strongest usable strength: {best:g}")
            print(f"    detection  TP {tp:.3f}  vs  FP {fp:.3f}   "
                  f"lift {tp - fp:+.3f}")
            print(f"    identification {idc:.3f}  vs chance 0.500   "
                  f"lift {idc - 0.5:+.3f}")
            summary["headline"] = {"fraction": best, "tp": tp, "fp": fp,
                                   "identify": idc, "n_usable": len(sub)}
    else:
        print("\n  NO usable non-zero strength: every injection that was strong enough")
        print("  to try also stopped the model answering. That is a reportable result.")

    path = RESULTS / f"injection__{slug(lm.label)}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {path.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
