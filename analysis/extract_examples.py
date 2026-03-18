#!/usr/bin/env python3
"""One-off extraction of C2b vs C4 content for example_segments doc."""
import json
import re
from pathlib import Path

def extract_ollama(raw):
    if not raw:
        return ""
    # Double-quoted content
    m = re.search(r'content="((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    if m:
        s = m.group(1)
        s = s.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        return s.strip()
    # Single-quoted content (e.g. content='...')
    m2 = re.search(r"content='((?:[^'\\]|\\.)*)'", raw, re.DOTALL)
    if m2:
        s = m2.group(1)
        s = s.replace('\\n', '\n').replace("\\'", "'").replace('\\\\', '\\')
        return s.strip()
    return ""

def main():
    base = Path("results/mts")
    c2b = json.load((base / "mts_c2b_results.json").open())
    c4_raw = json.load((base / "mts_c4_den_mat_results.json").open())
    cleaned = json.load((base / "cleaned" / "c4_den_mat_cleaned.json").open())
    c2b_by_id = {r["sample_id"]: r for r in c2b}

    for i, rec in enumerate(c4_raw):
        if i >= len(cleaned):
            break
        sid = rec["sample_id"]
        if sid not in [434, 599, 473, 371, 1067]:
            continue
        gt = rec.get("dialogue") or ""
        noisy = rec.get("noisy") or ""
        c4t = cleaned[i]
        c2b_rec = c2b_by_id.get(sid)
        if not c2b_rec:
            continue
        c2b_text = extract_ollama(c2b_rec.get("c2b_mistral") or "")
        print("=" * 60)
        print("Sample", sid)
        print("-" * 40)
        print("GT:", gt[:350])
        print("-" * 40)
        print("Noisy:", noisy[:250])
        print("-" * 40)
        print("C2b:", c2b_text[:400] if c2b_text else "N/A")
        print("-" * 40)
        print("C4:", c4t[:400] if isinstance(c4t, str) else str(c4t)[:400])
        print()

if __name__ == "__main__":
    main()
