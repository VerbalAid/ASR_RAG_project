import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any

import ollama


# ========= CONFIG =========
RESULTS_DIR = Path("results") / "ted"
INPUT_C1 = RESULTS_DIR / "c1_whisper_raw.json"
OUTPUT_C2_LLAMA = RESULTS_DIR / "c2_outputs_llama.json"

CHUNK_WORD_TARGET = 400
CHUNK_OVERLAP_WORDS = 50

LLAMA_MODEL_NAME = "llama3:8b"          # adjust to your local tag
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0


# ========= CHUNKING =========
def chunk_with_overlap(text: str,
                       chunk_size: int = CHUNK_WORD_TARGET,
                       overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


# ========= OLLAMA CALL =========
def safe_ollama_chat(messages: List[Dict[str, str]],
                     model: str = LLAMA_MODEL_NAME,
                     max_retries: int = MAX_RETRIES,
                     base_backoff: float = BASE_BACKOFF_SECONDS) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp: Any = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": 0.0, "top_p": 1.0},
            )
            if isinstance(resp, dict):
                msg = resp.get("message") or {}
                content = msg.get("content") or ""
            else:
                content = str(resp)
            if content:
                return content
        except Exception as e:
            print(f"[LLaMA] attempt {attempt}/{max_retries} failed: {e}")
        time.sleep(base_backoff * (2 ** (attempt - 1)))
    return ""


PREAMBLE_PATTERN = re.compile(
    r"^(here is( the)? corrected transcript:?|corrected text:?|"
    r"here's the correction:?|note:).*?\n",
    flags=re.IGNORECASE | re.DOTALL,
)


def clean_response(raw: str) -> str:
    if not raw:
        return ""
    text = raw.strip()
    text = PREAMBLE_PATTERN.sub("", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def correct_transcript_llama(asr_text: str) -> str:
    chunks = chunk_with_overlap(asr_text)
    if not chunks:
        return ""
    corrected = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"  Chunk {i}/{len(chunks)} ({len(chunk.split())} words)")
        user_instruction = (
            "You will receive a noisy ASR transcript excerpt.\n"
            "Fix only clear transcription errors (misrecognized words, obvious misspellings, incorrect names).\n"
            "Do NOT change meaning, summarise, or add commentary.\n"
            "Output ONLY the corrected text of the excerpt.\n\n"
            f"TRANSCRIPT EXCERPT:\n{chunk}"
        )
        messages = [{"role": "user", "content": user_instruction}]
        raw = safe_ollama_chat(messages)
        cleaned = clean_response(raw)
        corrected.append(cleaned)
        time.sleep(0.1)
    return " ".join(corrected).strip()


def main():
    if not INPUT_C1.exists():
        print(f"[ERROR] Input file not found: {INPUT_C1}")
        return
    with open(INPUT_C1, "r") as f:
        results = json.load(f)

    for idx, entry in enumerate(results):
        speaker = entry.get("speaker_id", f"sample_{idx}")
        asr = entry.get("c1_whisper_tiny") or ""
        print(f"\n=== [{idx+1}/{len(results)}] {speaker} ===")
        if not asr.strip():
            entry["c2_llm_only"] = None
            entry["c2_flag"] = "empty_asr"
            continue
        corrected = correct_transcript_llama(asr)
        in_wc = len(asr.split())
        out_wc = len(corrected.split()) if corrected else 0
        ratio = out_wc / in_wc if in_wc else 0.0
        print(f"  Word ratio: {in_wc} → {out_wc} (ratio={ratio:.2f})")
        if not corrected or ratio < 0.5:
            print("  WARNING: output too short; marking failure")
            entry["c2_llm_only"] = None
            entry["c2_flag"] = "too_short"
        else:
            entry["c2_llm_only"] = corrected
            entry["c2_flag"] = "success"
        OUTPUT_C2_LLAMA.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_C2_LLAMA, "w") as f_out:
            json.dump(results, f_out, indent=2)

    print(f"\nDone. Saved LLaMA outputs to {OUTPUT_C2_LLAMA}")


if __name__ == "__main__":
    main()

