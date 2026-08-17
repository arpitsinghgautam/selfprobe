"""Phase 1-3: elicit preferences under every prompt-level persona condition.

    .venv\\Scripts\\python.exe scripts\\01_elicit.py --model Qwen/Qwen2.5-7B-Instruct

Writes one JSON per (model, persona, template) into results/. Re-running skips
conditions already on disk unless --force, so an interrupted sweep resumes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from personaprobe import OUTCOMES, elicit_preference_matrix, load_model
from personaprobe.outcomes import outcomes_third_person_self
from personaprobe.elicit import TEMPLATES
from personaprobe.personas import PERSONAS, by_name

RESULTS = Path(__file__).resolve().parent.parent / "results"


def slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--personas", nargs="*", default=None, help="default: all")
    ap.add_argument("--templates", nargs="*", default=["prefer"], choices=list(TEMPLATES))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N outcomes")
    ap.add_argument("--base", action="store_true", help="treat as a base model (no chat template)")
    ap.add_argument("--quant", choices=["4bit", "8bit"], default=None,
                    help="bitsandbytes quantisation; results are keyed separately so a "
                         "quantised run cannot overwrite a full-precision one")
    ap.add_argument("--third-person-self", action="store_true",
                    help="rewrite the self category in third person, holding content fixed; "
                         "separates 'the model has a stake' from 'the prompt says you'")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    base_outcomes = outcomes_third_person_self() if args.third_person_self else OUTCOMES
    outcomes = base_outcomes[: args.limit] if args.limit else base_outcomes
    personas = [by_name(p) for p in args.personas] if args.personas else PERSONAS

    print(f"model={args.model}  outcomes={len(outcomes)}  "
          f"pairs={len(outcomes) * (len(outcomes) - 1) // 2}")

    lm = load_model(args.model, is_chat=False if args.base else None, quant=args.quant)
    # Keyed separately so the third-person run cannot overwrite the main one.
    if args.third_person_self:
        lm.label = f"{lm.label}-3p"
    print(f"loaded: {lm.n_layers} layers, d_model={lm.d_model}, "
          f"chat={lm.is_chat}, supports_system={lm.supports_system}"
          + (f", quant={args.quant}" if args.quant else ""))

    # Record verbatim what the model is actually shown under each condition.
    # Chat templates can inject a default system prompt when none is supplied
    # (Qwen2.5 does), which would make a "no system prompt" condition something
    # materially different from its name — and would quietly change how the
    # baseline comparison should be read. Dumped for the appendix rather than
    # assumed.
    probe = TEMPLATES[args.templates[0]].format(a=outcomes[0].text, b=outcomes[1].text)
    exemplars = {p.name: lm.format(probe, p.system) for p in personas}
    (RESULTS / f"prompts__{slug(lm.label)}.json").write_text(
        json.dumps({"supports_system": lm.supports_system, "is_chat": lm.is_chat,
                    "exemplars": exemplars}, indent=2))
    print(f"  wrote prompt exemplars for {len(exemplars)} conditions")

    for template in args.templates:
        for persona in personas:
            tag = f"{slug(lm.label)}__{persona.name}__{template}"
            path = RESULTS / f"{tag}.json"
            if path.exists() and not args.force:
                print(f"  skip {tag} (exists)")
                continue

            t0 = time.time()
            res = elicit_preference_matrix(
                lm, outcomes, persona, template=template, batch_size=args.batch_size
            )
            res.meta["seconds"] = round(time.time() - t0, 1)
            res.meta["condition"] = "prompt_level"
            res.meta["persona_kind"] = persona.kind
            path.write_text(json.dumps(res.to_dict(), indent=2))

            iu_bias = res.order_bias[res.order_bias > 0].mean() if res.order_bias.any() else 0.0
            print(f"  {tag}: {res.meta['n_forward_passes']} passes in "
                  f"{res.meta['seconds']}s, order bias {iu_bias:.3f}")

    print(f"\nwrote to {RESULTS}")


if __name__ == "__main__":
    main()
