"""Build PriMock57 RAG corpus: consultations/*.txt, primock57_passages.json, README."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import textgrid


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "datasets" / "primock57_raw"
TRANSCRIPTS_DIR = RAW_ROOT / "transcripts"
NOTES_DIR = RAW_ROOT / "notes"

OUT_ROOT = ROOT / "results" / "primock57"
OUT_CONSULTS = OUT_ROOT / "consultations"
OUT_PASSAGES = OUT_ROOT / "primock57_passages.json"
OUT_README = OUT_ROOT / "README.md"


TAG_PATTERN = re.compile(r"</?UNSURE>|<UNIN/>|<INAUDIBLE_SPEECH/>")


@dataclass
class ConsultationData:
    source_id: str
    transcript: str
    note: str


def clean_text(s: str) -> str:
    s = TAG_PATTERN.sub("", s or "")
    return re.sub(r"\s+", " ", s).strip()


def parse_textgrid(path: Path) -> List[Dict[str, object]]:
    tg = textgrid.TextGrid()
    tg.read(str(path))
    utterances: List[Dict[str, object]] = []
    for tier in tg.tiers:
        for interval in tier.intervals:
            mark = clean_text(interval.mark)
            if mark:
                utterances.append(
                    {"text": mark, "from": float(interval.minTime), "to": float(interval.maxTime)}
                )
    return utterances


def build_dialogue(transcript_id: str) -> str:
    doc_path = TRANSCRIPTS_DIR / f"{transcript_id}_doctor.TextGrid"
    pat_path = TRANSCRIPTS_DIR / f"{transcript_id}_patient.TextGrid"
    if not doc_path.exists() or not pat_path.exists():
        raise FileNotFoundError(f"Missing transcript pair for {transcript_id}")

    doc = parse_textgrid(doc_path)
    pat = parse_textgrid(pat_path)
    for u in doc:
        u["speaker"] = "Doctor"
    for u in pat:
        u["speaker"] = "Patient"
    merged = doc + pat
    merged.sort(key=lambda x: x["from"])
    lines = [f"{u['speaker']}: {u['text']}" for u in merged]
    return "\n".join(lines).strip()


def load_note(path: Path) -> str:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return clean_text(obj.get("note", ""))


def chunk_words(text: str, size: int = 180, overlap: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    out: List[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = words[start : start + size]
        if len(chunk) < 20:
            continue
        out.append(" ".join(chunk))
        if start + size >= len(words):
            break
    return out


def collect_consultations() -> List[ConsultationData]:
    note_files = sorted(NOTES_DIR.glob("*.json"))
    rows: List[ConsultationData] = []
    for nf in note_files:
        source_id = nf.stem  # e.g., day1_consultation01
        note = load_note(nf)
        transcript = build_dialogue(source_id)
        rows.append(ConsultationData(source_id=source_id, transcript=transcript, note=note))
    return rows


def write_consultation_texts(rows: List[ConsultationData]) -> None:
    OUT_CONSULTS.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows, start=1):
        idx = f"{i:02d}"
        (OUT_CONSULTS / f"consultation_{idx}_transcript.txt").write_text(
            row.transcript, encoding="utf-8"
        )
        (OUT_CONSULTS / f"consultation_{idx}_note.txt").write_text(row.note, encoding="utf-8")


def build_passages(rows: List[ConsultationData]) -> List[Dict[str, object]]:
    passages: List[Dict[str, object]] = []
    for row in rows:
        for passage in chunk_words(row.note, size=180, overlap=30):
            passages.append({"source_consultation_id": row.source_id, "passage": passage})
    return passages


def write_readme(rows: List[ConsultationData], passages: List[Dict[str, object]]) -> None:
    lengths = [len(p["passage"].split()) for p in passages]
    avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
    readme = f"""# PriMock57 corpus preparation

This folder contains processed PriMock57 assets for medical-domain retrieval.

## Source

- Dataset: [babylonhealth/primock57](https://github.com/babylonhealth/primock57)
- License in source repo: Apache-2.0 (verify before publication/distribution)

## What was produced

- `consultations/consultation_XX_transcript.txt` (57 files)
- `consultations/consultation_XX_note.txt` (57 files)
- `primock57_passages.json` (note chunks for RAG)

## Processing steps

1. Loaded all note JSON files from `datasets/primock57_raw/notes`.
2. Built each consultation dialogue from paired doctor/patient TextGrid files in `datasets/primock57_raw/transcripts`.
3. Wrote one transcript and one note text file per consultation.
4. Chunked note text into 180-word passages with 30-word overlap.
5. Saved passages as JSON entries with `source_consultation_id`.

## Statistics

- Consultations processed: {len(rows)}
- Passages generated: {len(passages)}
- Average passage length (words): {avg_len:.1f}

## Citation

Papadopoulos Korfiatis, A., Moramarco, F., Sarac, R., & Savkov, A. (2022).
PriMock57: A Dataset Of Primary Care Mock Consultations.
In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)* (pp. 588–598).
Association for Computational Linguistics.
https://doi.org/10.18653/v1/2022.acl-short.65
"""
    OUT_README.write_text(readme, encoding="utf-8")


def main() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Missing dataset directory: {RAW_ROOT}")
    rows = collect_consultations()
    if not rows:
        raise RuntimeError("No consultations found in PriMock57.")
    write_consultation_texts(rows)
    passages = build_passages(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_PASSAGES.write_text(json.dumps(passages, indent=2), encoding="utf-8")
    write_readme(rows, passages)
    print(f"Prepared {len(rows)} consultations.")
    print(f"Saved passages: {OUT_PASSAGES} ({len(passages)} entries)")
    print(f"README: {OUT_README}")


if __name__ == "__main__":
    main()
