"""Project 2, elicitation: stated ratings and predicted choices.

Produces two measurements to set against the revealed pairwise preferences
already collected by 01_elicit.py:

  ratings      — each outcome rated on its own, both scale directions (80 passes)
  predictions  — predicted choice of "an AI assistant" / "a different AI
                 assistant" over the same pairs (1560 passes each)

    .venv\\Scripts\\python.exe scripts\\08_stated.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from personaprobe import OUTCOMES, load_model
from personaprobe.personas import by_name
from personaprobe.rating import TARGETS, elicit_predictions, elicit_ratings

RESULTS = Path(__file__).resolve().parent.parent / "results"


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--personas", nargs="*", default=["default"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--skip-predictions", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    outcomes = OUTCOMES
    lm = load_model(args.model, is_chat=False if args.base else None)
    print(f"loaded {args.model}: supports_system={lm.supports_system}")

    for pname in args.personas:
        persona = by_name(pname)

        # --- Stated: cardinal ratings ---------------------------------------
        t0 = time.time()
        r = elicit_ratings(lm, outcomes, persona, batch_size=args.batch_size)
        r.meta["seconds"] = round(time.time() - t0, 1)
        (RESULTS / f"ratings__{slug(args.model)}__{pname}.json").write_text(
            json.dumps(r.to_dict(), indent=2))
        flag = "" if r.is_valid else "   <-- INVALID"
        print(f"  ratings/{pname}: mass {r.meta['scale_mass_mean']:.3f}, "
              f"sd {r.meta['rating_sd']:.3f}, scale disagreement "
              f"{r.scale_disagreement:.3f}{flag}")

        if args.skip_predictions:
            continue

        # --- Predicted choices ----------------------------------------------
        for label, (tmpl, target) in TARGETS.items():
            t0 = time.time()
            P, meta = elicit_predictions(lm, outcomes, persona, target=target,
                                         template=tmpl, batch_size=args.batch_size)
            meta["target_label"] = label
            meta["seconds"] = round(time.time() - t0, 1)
            (RESULTS / f"predict__{slug(args.model)}__{pname}__{label}.json").write_text(
                json.dumps({"outcome_ids": [o.id for o in outcomes],
                            "P": P.tolist(), "meta": meta,
                            "model": args.model, "persona": pname,
                            "target_label": label}, indent=2))
            print(f"  predict/{pname}/{label}: mass {meta['ab_mass_mean']:.3f}, "
                  f"order bias {meta['order_bias_mean']:.3f}, {meta['seconds']}s")

    print(f"\nwrote to {RESULTS}")


if __name__ == "__main__":
    main()
