"""Check a draft against the formatting rules before it goes out.

Covers the mechanical rules that can be checked automatically. The em dash and
colon rules are zero tolerance, so those are reported verbatim with line numbers.
Judgement calls like whether a subsection is earned are left to a human read.

    .venv\\Scripts\\python.exe scripts\\22_format_check.py report/report_4page.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Paths, filenames and function names in running prose read as a build log
# rather than as a person explaining the work.
BREADCRUMB = re.compile(r"\b\w+\.(?:py|json|parquet|csv|ps1|docx)\b|\bscripts/|\bresults/|\breport/")
SWEEPING = re.compile(r"\b(?:all prior work|current practice|every study|no prior work|"
                      r"nobody (?:else )?(?:reports|checks|does)|not standard in this literature|"
                      r"never reported|no one reports)\b", re.I)


def in_code_or_table(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") or s.startswith("```") or s.startswith("    ")


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "report/report_4page.md")
    lines = path.read_text(encoding="utf-8").splitlines()
    text = "\n".join(lines)
    fails = 0

    print(f"\n=== {path.name} ===\n")

    # A1. Em dashes, zero tolerance.
    hits = [(i + 1, l) for i, l in enumerate(lines) if "—" in l]
    print(f"A1  em dashes           {len(hits)} found")
    for n, l in hits[:12]:
        print(f"      line {n}: {l.strip()[:96]}")
    fails += len(hits)

    # A2. Colons in prose. Tables, code and headings are exempt.
    colon = []
    for i, l in enumerate(lines):
        if in_code_or_table(l) or l.strip().startswith("#"):
            continue
        stripped = re.sub(r"https?://\S+", "", l)
        stripped = re.sub(r"\b\d+:\d+\b", "", stripped)
        if ":" in stripped:
            colon.append((i + 1, l))
    print(f"\nA2  colons in prose     {len(colon)} found")
    for n, l in colon[:12]:
        print(f"      line {n}: {l.strip()[:96]}")
    fails += len(colon)

    # A3. Over-long sentences.
    body = "\n".join(l for l in lines if not in_code_or_table(l) and not l.strip().startswith("#"))
    long_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body)
                  if len(s.split()) > 35]
    print(f"\nA3  sentences over 35w  {len(long_sents)} found")
    for s in long_sents[:5]:
        print(f"      ({len(s.split())}w) {s[:100]}...")

    # C7. Emphasis in body prose.
    bold = [(i + 1, m) for i, l in enumerate(lines) if not in_code_or_table(l)
            for m in re.findall(r"\*\*(.+?)\*\*", l)]
    print(f"\nC7  bold spans in prose {len(bold)}")

    # N50. Reproduction breadcrumbs.
    crumbs = [(i + 1, l) for i, l in enumerate(lines)
              if not in_code_or_table(l) and not l.strip().startswith("-")
              and BREADCRUMB.search(l)]
    print(f"\nN50 breadcrumbs in prose {len(crumbs)} found")
    for n, l in crumbs[:8]:
        print(f"      line {n}: {l.strip()[:96]}")

    # N51. Sweeping prior-practice claims.
    sweep = [(i + 1, l) for i, l in enumerate(lines) if SWEEPING.search(l)]
    print(f"\nN51 sweeping claims     {len(sweep)} found")
    for n, l in sweep[:8]:
        print(f"      line {n}: {l.strip()[:96]}")

    # E10. Abstract length.
    m = re.search(r"## Abstract\n(.*?)\n## ", text, re.S)
    if m:
        abstract = re.sub(r"\*\(.*?\)\*", "", m.group(1), flags=re.S)
        print(f"\nE10 abstract            {len(abstract.split())} words (target 150-250)")
        print(f"E11 citations in abstract {'FOUND' if re.search(r'\\(\\d{4}\\)|et al', abstract) else 'none'}")

    # B5. Subsection depth.
    deep = [l for l in lines if re.match(r"^#### |^### \d+\.\d+\.\d+", l)]
    print(f"\nB5  headings past 1 level {len(deep)}")

    # D8. Outstanding placeholders.
    for tag in ("[CITE:", "[VERIFY:"):
        found = [(i + 1, l.strip()) for i, l in enumerate(lines) if tag in l]
        print(f"\nD   {tag:<9} placeholders {len(found)}")
        for n, l in found[:6]:
            print(f"      line {n}: {l[:96]}")

    print(f"\n{'ZERO-TOLERANCE FAILURES: ' + str(fails) if fails else 'No em dashes or prose colons.'}")


if __name__ == "__main__":
    main()
