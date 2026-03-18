#!/usr/bin/env python3
"""Print thesis-relevant values read from the codebase (noise params, paths, etc.)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    lines = []
    lines.append("=" * 60)
    lines.append("THESIS FACTS FROM CODEBASE")
    lines.append("=" * 60)

    # --- MTS noise (experiments/mts/noise.py) ---
    from experiments.mts import noise
    lines.append("")
    lines.append("--- MTS noise (§3.2) ---")
    lines.append(f"  WORD_DROP_P   = {noise.WORD_DROP_P} ({noise.WORD_DROP_P*100:.0f}% word drops)")
    lines.append(f"  WORD_SWAP_P   = {noise.WORD_SWAP_P} ({noise.WORD_SWAP_P*100:.0f}% word swaps)")
    lines.append(f"  CHAR_SUB_P    = {noise.CHAR_SUB_P} ({noise.CHAR_SUB_P*100:.0f}% char substitutions)")

    # --- Cleaned file name ---
    cleaned_dir = ROOT / "results" / "mts" / "cleaned"
    cleaned_json = cleaned_dir / "c4_den_mat_cleaned.json"
    c4b_cleaned = cleaned_dir / "c4b_cleaned.json"
    lines.append("")
    lines.append("--- C4-Den-Mat cleaned file (§4.1) ---")
    lines.append(f"  c4_den_mat_cleaned.json exists: {cleaned_json.exists()}")
    lines.append(f"  c4b_cleaned.json exists:        {c4b_cleaned.exists()}")

    # --- MTS C4b LOO vs all notes (experiments/mts/run.py) ---
    run_py = ROOT / "experiments" / "mts" / "run.py"
    text = run_py.read_text()
    has_c4b_all = "clinical_notes = [ds[i][\"section_text\"] for i in idxs" in text or 'for i in idxs' in text
    has_c3_mat_loo = "j != i" in text and "other_notes" in text
    lines.append("")
    lines.append("--- MTS C4-Den-Mat corpus (§4.1) ---")
    lines.append("  C4b: corpus from all idxs. C3-Lex-Mat: other_notes with j != i (LOO).")

    # --- Chunking ---
    from experiments.mts.utils import CHUNK_WORD_TARGET, CHUNK_OVERLAP_WORDS
    lines.append("")
    lines.append("--- Chunking (§4) ---")
    lines.append(f"  CHUNK_WORD_TARGET   = {CHUNK_WORD_TARGET}")
    lines.append(f"  CHUNK_OVERLAP_WORDS = {CHUNK_OVERLAP_WORDS}")

    # --- Models ---
    from experiments.mts.utils import MISTRAL_MODEL, LLAMA_MODEL
    lines.append("")
    lines.append("--- Models ---")
    lines.append(f"  MISTRAL_MODEL = {MISTRAL_MODEL}")
    lines.append(f"  LLAMA_MODEL   = {LLAMA_MODEL}")

    # --- Ground truth tags (prepare_primock57, c1_whisper) ---
    prep = ROOT / "analysis" / "prepare_primock57.py"
    tag_line = None
    if prep.exists():
        for line in prep.read_text().splitlines():
            if "TAG_PATTERN" in line and "re.compile" in line:
                tag_line = line.strip()
                break
    if tag_line:
        lines.append("")
        lines.append("--- PriMock57 ground-truth tags stripped (§3.3) ---")
        lines.append(f"  {tag_line}")

    # --- BLEU smoothing ---
    mts_eval = ROOT / "analysis" / "mts_eval.py"
    if mts_eval.exists():
        content = mts_eval.read_text()
        if "SmoothingFunction" in content and "method1" in content:
            lines.append("")
            lines.append("--- BLEU smoothing (Table 2) ---")
            lines.append("  SmoothingFunction().method1")

    # --- Condition count ---
    lines.append("")
    lines.append("--- Condition count (Abstract / §1) ---")
    lines.append("  C1 + C2a + C2b + 6 RAG = 9 conditions.")

    # --- NER: MTS vs PriMock57 ---
    lines.append("")
    lines.append("--- NER (Table 2 / §6.7) ---")
    lines.append("  NER F1: MTS-Dialog and PriMock57 (scispaCy CHEM/DIS).")

    # --- PriMock57 passage count ---
    primock57_passages = ROOT / "results" / "primock57" / "primock57_passages.json"
    if primock57_passages.exists():
        import json
        data = json.loads(primock57_passages.read_text())
        n_passages = len(data) if isinstance(data, list) else len(data.get("passages", []))
        lines.append("")
        lines.append("--- PriMock57 passages (MTS C3-Lex-Rel / C4-Den-Rel) ---")
        lines.append(f"  primock57_passages.json: {n_passages} entries.")

    lines.append("")
    lines.append("=" * 60)
    out = "\n".join(lines)
    print(out)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
