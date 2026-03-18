"""Transcribe first N PriMock57 consultations with Whisper-tiny; writes c1_whisper_raw.json."""
import json
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import whisper

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "datasets" / "primock57_raw"
AUDIO_DIR = RAW_ROOT / "audio_consultations"
if not AUDIO_DIR.exists():
    AUDIO_DIR = RAW_ROOT / "audio consultations"
if not AUDIO_DIR.exists():
    AUDIO_DIR = RAW_ROOT / "audio"
TRANSCRIPTS_DIR = RAW_ROOT / "transcripts"
OUT_DIR = ROOT / "results" / "primock57_speech"
OUT_FILE = OUT_DIR / "c1_whisper_raw.json"

NUM_CONSULTATIONS = 10
WHISPER_MODEL = "tiny"

TAG_PATTERN = re.compile(r"</?UNSURE>|<UNIN/>|<INAUDIBLE_SPEECH/>")


def clean_text(s: str) -> str:
    s = TAG_PATTERN.sub("", s or "")
    return re.sub(r"\s+", " ", s).strip()


def parse_primock57_textgrid(path: Path) -> list:
    """
    Parse PriMock57 Praat TextGrid: read file and extract each interval's
    xmin, xmax, and text (the correct transcription). Format:
      intervals [N]:
          xmin = ...
          xmax = ...
          text = "..." or text = ""
    """
    raw = path.read_text(encoding="utf-8")
    out = []
    # Split into interval blocks (header before first "intervals [" is ignored)
    blocks = re.split(r"\s*intervals\s*\[\d+\]\s*:\s*", raw)
    for block in blocks[1:]:
        xmin = xmax = None
        text = ""
        for line in block.split("\n"):
            line_strip = line.strip()
            if line_strip.startswith("xmin ="):
                xmin = float(line_strip.split("=", 1)[1].strip())
            elif line_strip.startswith("xmax ="):
                xmax = float(line_strip.split("=", 1)[1].strip())
            elif line_strip.startswith("text ="):
                rest = line_strip.split("=", 1)[1].strip()
                if rest.startswith('"') and rest.endswith('"'):
                    text = rest[1:-1].replace('\\"', '"')
                else:
                    text = rest.strip('"')
        if xmin is not None and xmax is not None:
            cleaned = clean_text(text)
            if cleaned:
                out.append({"text": cleaned, "from": xmin, "to": xmax})
    return out


def build_dialogue(transcript_id: str) -> str:
    """Ground truth: time-ordered Doctor/Patient dialogue from TextGrids in primock57_raw/transcripts."""
    doc_path = TRANSCRIPTS_DIR / f"{transcript_id}_doctor.TextGrid"
    pat_path = TRANSCRIPTS_DIR / f"{transcript_id}_patient.TextGrid"
    if not doc_path.exists() or not pat_path.exists():
        raise FileNotFoundError(f"Missing transcript pair for {transcript_id}")
    doc = parse_primock57_textgrid(doc_path)
    pat = parse_primock57_textgrid(pat_path)
    for u in doc:
        u["speaker"] = "Doctor"
    for u in pat:
        u["speaker"] = "Patient"
    merged = doc + pat
    merged.sort(key=lambda x: x["from"])
    return "\n".join(f"{u['speaker']}: {u['text']}" for u in merged).strip()


def get_first_n_consultation_ids(n: int = NUM_CONSULTATIONS) -> list[str]:
    """Same order as prepare_primock57: sorted note file stems."""
    notes_dir = RAW_ROOT / "notes"
    if not notes_dir.exists():
        raise FileNotFoundError(f"Missing {notes_dir}. Run from project root.")
    note_files = sorted(notes_dir.glob("*.json"))
    return [nf.stem for nf in note_files[:n]]


def load_and_concat_audio(consultation_id: str) -> Tuple[np.ndarray, int]:
    """Load doctor and patient wav, concatenate (doctor then patient), return (samples, sr)."""
    doc_path = AUDIO_DIR / f"{consultation_id}_doctor.wav"
    pat_path = AUDIO_DIR / f"{consultation_id}_patient.wav"
    if not doc_path.exists() or not pat_path.exists():
        raise FileNotFoundError(f"Missing audio for {consultation_id}")
    doc_audio, sr_d = sf.read(doc_path, dtype="float32")
    pat_audio, sr_p = sf.read(pat_path, dtype="float32")
    if sr_d != sr_p:
        raise RuntimeError(f"Sample rate mismatch {consultation_id}: {sr_d} vs {sr_p}")
    if doc_audio.ndim > 1:
        doc_audio = doc_audio.mean(axis=1)
    if pat_audio.ndim > 1:
        pat_audio = pat_audio.mean(axis=1)
    combined = np.concatenate([doc_audio, pat_audio])
    return combined, sr_d


def main():
    if not RAW_ROOT.exists():
        print(f"Error: PriMock57 raw dataset not found at {RAW_ROOT}", file=sys.stderr)
        sys.exit(1)

    ids = get_first_n_consultation_ids(NUM_CONSULTATIONS)
    print(f"Transcribing first {NUM_CONSULTATIONS} consultations: {ids}")

    model = whisper.load_model(WHISPER_MODEL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for i, consultation_id in enumerate(ids):
        print(f"  [{i+1}/{len(ids)}] {consultation_id}...")
        try:
            ground_truth = build_dialogue(consultation_id)
        except Exception as e:
            print(f"    Skip: {e}")
            continue
        try:
            audio, sr = load_and_concat_audio(consultation_id)
        except Exception as e:
            print(f"    Skip: {e}")
            continue

        out = model.transcribe(audio, language="en")
        raw_text = (out.get("text") or "").strip()

        results.append({
            "sample_id": i,
            "speaker_id": consultation_id,
            "ground_truth": ground_truth,
            "c1_whisper_tiny": raw_text,
        })

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} samples to {OUT_FILE}")


if __name__ == "__main__":
    main()
