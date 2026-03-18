"""Parse raw MTS C4b outputs into cleaned transcript strings (c4_den_mat_cleaned.json, optional txt)."""

import json
import re
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "mts"
RAW_DIR = RESULTS_DIR / "raw"
C4_DEN_MAT_RESULTS_PATH = RESULTS_DIR / "mts_c4_den_mat_results.json"
CLEANED_DIR = RESULTS_DIR / "cleaned"
TXT_DIR = CLEANED_DIR / "txt"


def extract_ollama_content(raw: str) -> str:
    """
    Extract assistant message content from an Ollama response string.
    Handles content="..." with escaped quotes (\\") and content='...' with escaped quotes.
    """
    if not raw or not isinstance(raw, str):
        return ""
    # Double-quoted content: match content=" then any sequence of (non-quote or \") until ", thinking
    m = re.search(r'content="((?:[^"\\]|\\.)*)"\s*,\s*thinking', raw, re.DOTALL)
    if m:
        return m.group(1).replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\").strip()
    # Single-quoted content (e.g. content=' ... \\' ... ')
    m2 = re.search(r"content='((?:[^'\\]|\\.)*)'\s*,\s*thinking", raw, re.DOTALL)
    if m2:
        return m2.group(1).replace("\\'", "'").replace("\\n", "\n").strip()
    return ""


def main() -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    txt_out = TXT_DIR.exists() or True  # write txt by default
    if txt_out:
        TXT_DIR.mkdir(parents=True, exist_ok=True)

    contents: list[str] = []
    failures: list[str] = []
    source = "unknown"

    # Prefer per-dialogue JSON files in raw/
    raw_files = sorted(RAW_DIR.glob("*.json")) if RAW_DIR.exists() else []
    if raw_files:
        source = "raw"
        for p in raw_files:
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                msg = data.get("message") or {}
                if isinstance(msg, dict):
                    text = msg.get("content") or ""
                else:
                    text = ""
                if not text:
                    failures.append(p.name)
                else:
                    contents.append(text.strip())
            except (json.JSONDecodeError, OSError) as e:
                failures.append(f"{p.name} ({e})")
    else:
        # Fallback: single mts_c4b_results.json
        if not C4_DEN_MAT_RESULTS_PATH.exists():
            print(f"[parse_mts_raw] No raw dir and {C4_DEN_MAT_RESULTS_PATH} not found. Exiting.")
            return
        source = "mts_c4_den_mat_results.json"
        with open(C4_DEN_MAT_RESULTS_PATH, "r") as f:
            records = json.load(f)
        for i, rec in enumerate(records):
            raw = rec.get("c4b_dense_clinical") or ""
            text = extract_ollama_content(raw)
            if not text:
                failures.append(f"record_{i+1}")
            contents.append(text if text else "")

    # Write JSON array
    out_json = CLEANED_DIR / "c4_den_mat_cleaned.json"
    with open(out_json, "w") as f:
        json.dump(contents, f, indent=2)
    print(f"[parse_mts_raw] Parsed from {source}: {len(contents)} dialogues, {len(failures)} failures.")
    if failures:
        print(f"  Failures: {failures[:10]}{'...' if len(failures) > 10 else ''}")
    print(f"  Saved to {out_json}")

    # Optional: one .txt per dialogue
    if txt_out and contents:
        for i, text in enumerate(contents):
            num = i + 1
            name = f"dialogue_{num:03d}.txt"
            (TXT_DIR / name).write_text(text, encoding="utf-8")
        print(f"  Wrote {len(contents)} files to {TXT_DIR}/")


if __name__ == "__main__":
    main()
