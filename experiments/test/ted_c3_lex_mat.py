"""
C3-Lex-Mat on TED: BM25 retrieval over leave-one-speaker-out TED ground-truth (domain-matched).

Same corpus as C4-Den-Mat: leave-one-speaker-out TED-LIUM ground truth. BM25Okapi, top-3.
Chunking: 400 words, 50 overlap. Mistral 7B via Ollama.
Output: results/ted/ted_c3_lex_mat_results.json
"""
import json
import re
import time
import logging
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi
import ollama

CHUNK_WORD_TARGET = 400
CHUNK_OVERLAP_WORDS = 50
BM25_TOP_N = 3
OLLAMA_MODEL = "mistral:latest"

RESULTS_PATH = Path("results")
TED_PATH = RESULTS_PATH / "ted"
INPUT_C1 = TED_PATH / "c1_whisper_raw.json"
OUTPUT_FILE = TED_PATH / "ted_c3_lex_mat_results.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("c3_lex_mat_ted")


def simple_sentence_split(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def chunk_transcript(text: str) -> List[str]:
    sentences = simple_sentence_split(text)
    chunks = []
    current = []
    current_words = 0

    for s in sentences:
        w = len(s.split())
        if current_words + w <= CHUNK_WORD_TARGET or not current:
            current.append(s)
            current_words += w
        else:
            chunks.append(" ".join(current))
            overlap_words = 0
            overlap_sentences = []
            for sent in reversed(current):
                overlap_sentences.insert(0, sent)
                overlap_words += len(sent.split())
                if overlap_words >= CHUNK_OVERLAP_WORDS:
                    break
            current = overlap_sentences.copy()
            current_words = overlap_words
            current.append(s)
            current_words += w

    if current:
        chunks.append(" ".join(current))
    return chunks


def build_bm25_for_speaker(results: list, test_speaker_id: str):
    """Build BM25 index from all TED ground-truth transcripts EXCEPT test speaker (domain-matched LOO)."""
    passages = []
    for r in results:
        if r.get("speaker_id") == test_speaker_id:
            continue
        gt = r.get("ground_truth", "").strip()
        if not gt:
            continue
        words = gt.split()
        for i in range(0, len(words), 200):
            chunk = words[i : i + 200]
            if len(chunk) >= 30:
                passages.append(" ".join(chunk))
    tokenized = [tokenize(p) for p in passages]
    bm25 = BM25Okapi(tokenized)
    return passages, bm25


OLLAMA_TIMEOUT = 300

def safe_ollama_chat(messages, max_retries=3, pause=1.0) -> str:
    client = ollama.Client(timeout=OLLAMA_TIMEOUT)
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options={"temperature": 0, "num_predict": 768},
            )
            if isinstance(resp, dict):
                content = resp.get("message", {}).get("content") or resp.get("text") or ""
            else:
                content = str(resp)
            if content:
                return content
        except Exception as e:
            log.warning(f"Ollama attempt {attempt} failed: {e}")
        time.sleep(pause * (2 ** (attempt - 1)))
    return ""


def clean_response(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^(corrected text:|here is the corrected transcript:)", "", text, flags=re.I)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    TED_PATH.mkdir(parents=True, exist_ok=True)
    if not INPUT_C1.exists():
        log.error(f"Input not found: {INPUT_C1}")
        return

    with INPUT_C1.open("r") as f:
        results = json.load(f)

    for entry in results:
        speaker = entry.get("speaker_id", "unknown")
        log.info(f"Processing speaker: {speaker}")

        raw_asr = entry.get("c1_whisper_tiny", "")
        if not raw_asr:
            log.warning(f"No ASR text for {speaker}; skipping.")
            continue

        passages, bm25 = build_bm25_for_speaker(results, speaker)
        log.info(f"  Corpus: {len(passages)} passages (leave-one-out for {speaker})")

        chunks = chunk_transcript(raw_asr)
        corrected_chunks = []

        for i, chunk in enumerate(chunks, start=1):
            log.info(f"  Chunk {i}/{len(chunks)}")
            query_tokens = tokenize(chunk)
            top_passages = bm25.get_top_n(query_tokens, passages, n=BM25_TOP_N)
            context = "\n\n".join(top_passages)

            prompt = (
                f"REFERENCE CONTEXT:\n{context}\n\n"
                f"TRANSCRIPT:\n{chunk}\n\n"
                "Fix only clear ASR errors. Output ONLY corrected text."
            )
            messages = [{"role": "user", "content": prompt}]
            raw_resp = safe_ollama_chat(messages)
            cleaned = clean_response(raw_resp)
            corrected_chunks.append(cleaned)
            time.sleep(0.25)

        final_output = " ".join(corrected_chunks)
        input_wc = len(raw_asr.split())
        output_wc = len(final_output.split())
        ratio = (output_wc / input_wc) if input_wc else 0.0
        log.info(f"  Word ratio: {ratio:.2f}")

        if ratio < 0.5:
            entry["c3_lex_mat_bm25_rag"] = None
            entry["c3_lex_mat_flag"] = "too_short"
        else:
            entry["c3_lex_mat_bm25_rag"] = final_output
            entry["c3_lex_mat_flag"] = "success"

        with OUTPUT_FILE.open("w") as f:
            json.dump(results, f, indent=2)

    log.info(f"C3-Lex-Mat TED run complete. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
