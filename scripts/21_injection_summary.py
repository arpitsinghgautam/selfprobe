"""Cross-model summary of the concept-injection benchmark.

Produces the table and figure for project 2's introspection section. The figure
plots true-positive rate against false-alarm rate side by side, because the whole
point is that they must be read together: one model in this set has the highest
detection rate and essentially no discrimination.

    .venv\\Scripts\\python.exe scripts\\21_injection_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DISPLAY = {
    "Qwen_Qwen2.5-0.5B-Instruct": ("Qwen 0.5B", 0.5),
    "Qwen_Qwen2.5-1.5B-Instruct": ("Qwen 1.5B", 1.5),
    "Qwen_Qwen2.5-3B-Instruct": ("Qwen 3B", 3.0),
    "Qwen_Qwen2.5-7B-Instruct": ("Qwen 7B", 7.0),
    "unsloth_Qwen2.5-14B-Instruct-bnb-4bit-prequantized": ("Qwen 14B", 14.0),
    "mistralai_Mistral-7B-Instruct-v0.3": ("Mistral 7B", 7.0),
    "microsoft_Phi-3.5-mini-instruct": ("Phi-3.5", 3.8),
    "tiiuae_Falcon3-7B-Instruct": ("Falcon3 7B", 7.0),
    "allenai_OLMo-2-1124-7B-Instruct": ("OLMo-2 7B", 7.0),
}
BANDS = ["early", "middle", "late"]


def main() -> None:
    rows = []
    for p in sorted(RESULTS.glob("injection__*.json")):
        key = p.name[len("injection__"):-len(".json")]
        name, params = DISPLAY.get(key, (key, 0.0))
        d = json.loads(p.read_text())
        h = d.get("headline")
        if not h:
            continue
        bands = d.get("by_band", {})
        best_band = max(bands, key=lambda b: bands[b]["tp"] - bands[b]["fp"]) if bands else "—"
        best_id = max(bands, key=lambda b: bands[b]["identify"]) if bands else "—"
        rows.append({
            "key": key, "name": name, "params": params,
            "fp": h["fp"], "tp": h["tp"], "lift": h["tp"] - h["fp"],
            "identify": h["identify"], "id_lift": h["identify"] - 0.5,
            "best_detect_band": best_band, "best_id_band": best_id,
            "bands": bands,
        })
    if not rows:
        raise SystemExit("no injection results found")

    rows.sort(key=lambda r: -r["lift"])

    hdr = (f"{'model':<12}{'FP':>7}{'TP':>7}{'lift':>8}"
           f"{'identify':>10}{'vs chance':>11}{'best depth':>12}")
    print(f"\n=== Concept-injection introspection ===\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<12}{r['fp']:>7.3f}{r['tp']:>7.3f}{r['lift']:>+8.3f}"
              f"{r['identify']:>10.3f}{r['id_lift']:>+11.3f}{r['best_detect_band']:>12}")

    print("\n  FP = says a concept is injected when nothing is. TP = says so when one is.")
    print("  Identification is 2AFC against every other concept, both orders; chance 0.500.")

    worst = max(rows, key=lambda r: r["fp"])
    print(f"\n  Highest false-alarm rate: {worst['name']} at {worst['fp']:.3f}. "
          f"Its TP of {worst['tp']:.3f} is the")
    print(f"  highest in the table, and its discrimination is {worst['lift']:+.3f}. "
          f"Reporting TP alone")
    print(f"  would rank it first; the false-alarm control ranks it last.")

    # --- figure ------------------------------------------------------------
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    ax = axes[0]
    y = np.arange(len(rows))
    ax.barh(y + 0.19, [r["tp"] for r in rows], height=0.36,
            color="#2166ac", label="detects when injected (TP)")
    ax.barh(y - 0.19, [r["fp"] for r in rows], height=0.36,
            color="#b2182b", label="detects when NOT injected (FP)")
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("detection rate")
    ax.set_title("A high detection rate is not discrimination", loc="left", fontsize=10)
    # Below the axes: inside, the legend covered the two most important bars.
    ax.legend(frameon=False, fontsize=7.5, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    ax = axes[1]
    for r in rows:
        if not r["bands"]:
            continue
        vals = [r["bands"][b]["identify"] for b in BANDS if b in r["bands"]]
        xs = [BANDS.index(b) for b in BANDS if b in r["bands"]]
        ax.plot(xs, vals, marker="o", ms=3.5, lw=1.2, alpha=0.8, label=r["name"])
    ax.axhline(0.5, color="grey", lw=1, ls=":")
    ax.text(0.02, 0.503, "chance", fontsize=7.5, color="grey")
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(["early", "middle", "late"])
    ax.set_xlabel("injection depth")
    ax.set_ylabel("identification accuracy")
    ax.set_title("Introspective access peaks mid-network", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax.grid(alpha=0.25, lw=0.5)

    fig.text(0.5, -0.16,
             "Left: models ordered by discrimination (TP − FP). The two highest raw detection "
             "rates in the set belong to Qwen 0.5B (0.612) and\nFalcon3-7B (0.561) — and both "
             "have essentially zero discrimination, because they also report an injected concept "
             "62% and 55% of\nthe time when nothing is injected. Right: identification against "
             "every other concept, both orders. Middle-depth injection wins in\n7 of 8 models; "
             "late-layer injection is barely above chance.",
             ha="center", va="top", fontsize=6.5, color="#444")

    fig.tight_layout()
    out = FIGURES / "fig5_injection.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out.relative_to(ROOT)}")

    (RESULTS / "injection_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote results/injection_summary.json")


if __name__ == "__main__":
    main()
