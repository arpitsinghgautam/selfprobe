#!/usr/bin/env python3
"""
dblp_bibtex.py - upgrade references.bib to DBLP published-venue BibTeX.

For each entry in references.bib this:
  1. searches DBLP by title,
  2. picks the best-matching record, PREFERRING a published venue
     (conference / journal) over the arXiv (CoRR) preprint,
  3. downloads that record's BibTeX (condensed `?param=1` form),
  4. rewrites DBLP's citekey back to OUR original citekey so \\cite{} keeps working,
and writes references_dblp.bib plus a per-entry report.

DBLP rate-limits hard and 503s under load, so every HTTP call retries with
exponential backoff (default 8 tries). Run:  python dblp_bibtex.py
"""
import json, re, sys, time, difflib, urllib.parse, urllib.request, urllib.error
from pathlib import Path

HERE    = Path(__file__).resolve().parent
IN_BIB  = HERE / "references.bib"
OUT_BIB = HERE / "references_dblp.bib"

UA              = "Mozilla/5.0 (dblp-bibtex; mailto:arpitsinghgautam777@gmail.com)"
MAX_RETRIES     = 8
POLITE_SLEEP    = 3.0     # seconds between papers (be nice to DBLP; it rate-limits hard)
MATCH_THRESHOLD = 0.80    # title-similarity floor to accept a DBLP hit

# DBLP "type" buckets we treat as a real published venue (preferred over CoRR)
PUBLISHED_TYPES = {"Conference and Workshop Papers", "Journal Articles",
                   "Books and Theses", "Reference Works"}
ARXIV_VENUES    = {"corr", "arxiv"}


def http_get(url):
    """GET with retry + exponential backoff on transient errors (esp. 503)."""
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                wait = min(2 ** attempt, 30)
                print(f"      HTTP {e.code} (try {attempt}/{MAX_RETRIES}) -> wait {wait}s", file=sys.stderr)
                time.sleep(wait); continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            wait = min(2 ** attempt, 30)
            print(f"      {type(e).__name__} (try {attempt}/{MAX_RETRIES}) -> wait {wait}s", file=sys.stderr)
            time.sleep(wait); continue
    raise RuntimeError(f"failed after {MAX_RETRIES} retries: {url} ({last})")


def field_value(raw, field):
    """Extract a bib field's value (brace-balanced or quoted), anywhere in the entry."""
    m = re.search(r'(?i)\b' + field + r'\s*=\s*\{', raw)
    if m:
        i = m.end() - 1; depth = 0; j = i
        while j < len(raw):
            if raw[j] == '{': depth += 1
            elif raw[j] == '}':
                depth -= 1
                if depth == 0: break
            j += 1
        return re.sub(r'\s+', ' ', raw[i + 1:j]).strip()
    mq = re.search(r'(?i)\b' + field + r'\s*=\s*"(.*?)"', raw, re.S)
    return re.sub(r'\s+', ' ', mq.group(1)).strip() if mq else ""


def parse_entries(text):
    """Yield (citekey, entrytype, title, raw_entry) by brace-balancing each @entry."""
    out = []
    for m in re.finditer(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', text):
        etype, key = m.group(1), m.group(2).strip()
        i = text.index('{', m.start()); depth = 0; j = i
        while j < len(text):
            if text[j] == '{': depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0: break
            j += 1
        raw = text[m.start():j + 1]
        out.append((key, etype, field_value(raw, 'title'), raw))
    return out


def norm(s):
    return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()


def best_hit(our_title, hits):
    """Choose the best DBLP hit: title must match; prefer published venue over CoRR."""
    nt = norm(our_title)
    scored = []
    for h in hits:
        info = h.get("info", {})
        ht = info.get("title", "") or ""
        ratio = difflib.SequenceMatcher(None, nt, norm(ht)).ratio()
        if ratio < MATCH_THRESHOLD:
            continue
        venue = (info.get("venue") or "")
        venue_l = venue.lower() if isinstance(venue, str) else " ".join(map(str, venue)).lower()
        is_arxiv = any(a in venue_l for a in ARXIV_VENUES)
        published = (info.get("type") in PUBLISHED_TYPES) and not is_arxiv
        year = info.get("year", "0")
        # rank: published first, then has-doi, then newer, then title closeness
        scored.append((published, bool(info.get("doi")), year, round(ratio, 3), info))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]), reverse=True)
    return scored[0][4]


def fetch_bibtex(dblp_key):
    # condensed/standard bib form (matches the editor/booktitle/doi example)
    return http_get(f"https://dblp.org/rec/{dblp_key}.bib?param=1")


def retype_key(bibtext, our_key):
    """Replace DBLP's generated citekey with our original key (first @entry only)."""
    return re.sub(r'(@\w+\s*\{\s*)[^,]+,', lambda m: m.group(1) + our_key + ",", bibtext, count=1)


def main():
    text = IN_BIB.read_text(encoding="utf-8")
    entries = parse_entries(text)
    print(f"[dblp] {len(entries)} entries in {IN_BIB.name}\n")

    # Resume mode: `python dblp_bibtex.py key1 key2 ...` reprocesses ONLY those keys,
    # reusing the already-resolved entries from a prior references_dblp.bib for the rest.
    only = set(sys.argv[1:])
    prev = {k: r for (k, _, _, r) in parse_entries(OUT_BIB.read_text(encoding="utf-8"))} if (only and OUT_BIB.exists()) else {}
    if only:
        print(f"[dblp] resume mode: reprocessing only {sorted(only)}\n")

    out_chunks, report = [], []
    for idx, (key, etype, title, raw) in enumerate(entries, 1):
        print(f"[{idx}/{len(entries)}] {key}: {title[:60]}")
        if only and key not in only:
            out_chunks.append(prev.get(key, raw)); report.append((key, "reuse", "")); continue
        if not title:
            out_chunks.append(raw); report.append((key, "no-title", "kept original")); continue
        try:
            q = urllib.parse.urlencode({"q": title, "format": "json", "h": "20"})
            data = json.loads(http_get(f"https://dblp.org/search/publ/api?{q}"))
            hits = data.get("result", {}).get("hits", {}).get("hit", [])
            if isinstance(hits, dict):  # single hit -> dict not list
                hits = [hits]
            info = best_hit(title, hits)
            if not info:
                out_chunks.append(raw); report.append((key, "no-match", "kept original"))
                print("      no confident DBLP match -> kept original")
                time.sleep(POLITE_SLEEP); continue
            venue = info.get("venue", "?")
            venue = venue if isinstance(venue, str) else "/".join(map(str, venue))
            is_arxiv = venue.lower() in ARXIV_VENUES
            bib = retype_key(fetch_bibtex(info["key"]).strip(), key)
            out_chunks.append(bib)
            status = "ARXIV (CoRR)" if is_arxiv else f"PUBLISHED: {venue} {info.get('year','')}"
            report.append((key, status, info.get("doi", "")))
            print(f"      -> {status}")
        except Exception as e:
            out_chunks.append(raw); report.append((key, "ERROR", str(e)))
            print(f"      ERROR: {e} -> kept original", file=sys.stderr)
        time.sleep(POLITE_SLEEP)

    OUT_BIB.write_text("% Generated by dblp_bibtex.py - DBLP published-venue citations.\n\n"
                       + "\n\n".join(out_chunks) + "\n", encoding="utf-8")

    pub  = sum(1 for _, s, _ in report if s.startswith("PUBLISHED"))
    arx  = sum(1 for _, s, _ in report if s.startswith("ARXIV"))
    kept = sum(1 for _, s, _ in report if "kept original" in s)
    print(f"\n[dblp] wrote {OUT_BIB.name}: {pub} published, {arx} arXiv/CoRR, {kept} kept-original")
    print("\n=== REPORT ===")
    for key, status, extra in report:
        print(f"  {key:<22} {status}")


if __name__ == "__main__":
    main()
