# bm25_rag_simple.py
import json
import re
import time
import logging
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi
from datasets import load_dataset
import ollama

# ---------- CONFIG ----------
WIKI_PASSAGES_TARGET = 50000        # total passages to build (may stop early if dataset smaller)
WIKI_PASSAGE_WORD_MIN = 30
PASSAGE_WORD_TARGET = 200           # words per passage window

CHUNK_WORD_TARGET = 400
CHUNK_OVERLAP_WORDS = 50
BM25_TOP_N = 3

OLLAMA_MODEL = "mistral:latest"
OLLAMA_TIMEOUT = 300

RESULTS_PATH = Path("results")
TED_PATH = RESULTS_PATH / "ted"
WIKI_PATH = RESULTS_PATH / "wiki"
TED_PATH.mkdir(parents=True, exist_ok=True)
WIKI_PATH.mkdir(parents=True, exist_ok=True)

INPUT_C1 = TED_PATH / "c1_whisper_raw.json"
OUTPUT_FILE = TED_PATH / "c3_lex_gen_outputs_bm25.json"
CACHE_WIKI = WIKI_PATH / "c3_lex_gen_wikipedia_passages_cache.json"

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bm25_rag_simple")


# ---------- UTILITIES ----------
def simple_sentence_split(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def build_wiki_passages():
    """
    Minimal, deterministic loader:
    - If CACHE_WIKI exists, load it.
    - Otherwise stream English Wikipedia and slice into short passages.
    """
    if CACHE_WIKI.exists():
        log.info("Loading cached passages...")
        with open(CACHE_WIKI, "r") as f:
            return json.load(f)

    log.info("Building passages from Wikipedia...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    passages = []
    for item in ds:
        text = item.get("text") if isinstance(item, dict) else getattr(item, "text", "")
        if not text:
            continue

        # Keep the first few sentences as a compact retrieval passage.
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        passage = " ".join(sentences[:3]).strip()
        words = passage.split()
        if len(words) < WIKI_PASSAGE_WORD_MIN:
            continue

        for i in range(0, len(words), PASSAGE_WORD_TARGET):
            chunk = words[i : i + PASSAGE_WORD_TARGET]
            if len(chunk) < WIKI_PASSAGE_WORD_MIN:
                continue
            passages.append(" ".join(chunk))
            if len(passages) >= WIKI_PASSAGES_TARGET:
                break
        if len(passages) >= WIKI_PASSAGES_TARGET:
            break

    # cache for future runs
    log.info(f"Built {len(passages)} passages; caching to {CACHE_WIKI}")
    with open(CACHE_WIKI, "w") as f:
        json.dump(passages, f)

    return passages


def chunk_transcript(text: str) -> List[str]:
    sentences = simple_sentence_split(text)
    chunks = []
    current = []
    current_words = 0

    for s in sentences:
        w = len(s.split())
        # start new if would exceed target (unless empty)
        if current_words + w <= CHUNK_WORD_TARGET or not current:
            current.append(s)
            current_words += w
        else:
            chunks.append(" ".join(current))
            # create overlap from tail sentences
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


def safe_ollama_chat(messages, max_retries=3, pause=1.0):
    """
    Minimal retry wrapper for ollama.chat. Returns string (may be empty on failure).
    Uses a timeout so one slow request doesn't hang the run forever.
    """
    client = ollama.Client(timeout=OLLAMA_TIMEOUT)
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat(model=OLLAMA_MODEL, messages=messages, options={"temperature": 0, "num_predict": 768})
            # robust extraction: try common shapes
            if isinstance(resp, dict):
                content = resp.get("message", {}).get("content") or resp.get("text") or ""
            else:
                content = str(resp)
            if content:
                return content
        except Exception as e:
            log.warning(f"Ollama call attempt {attempt} failed: {e}")
        time.sleep(pause * (2 ** (attempt - 1)))
    return ""


def clean_response(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'^(corrected text:|here is the corrected transcript:)', '', text, flags=re.I)
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------- MAIN ----------
def main():
    # build/load passages and index BM25
    wiki_passages = build_wiki_passages()
    tokenized_corpus = [tokenize(p) for p in wiki_passages]
    log.info("Indexing BM25...")
    bm25 = BM25Okapi(tokenized_corpus)

    # load C1 baseline outputs
    if not INPUT_C1.exists():
        log.error(f"Input file not found: {INPUT_C1}")
        return

    with open(INPUT_C1, "r") as f:
        results = json.load(f)

    for entry in results:
        speaker = entry.get("speaker_id", "unknown")
        log.info(f"Processing speaker: {speaker}")

        raw_asr = entry.get("c1_whisper_tiny", "")
        if not raw_asr:
            log.warning(f"No ASR text for speaker {speaker}; skipping.")
            continue

        chunks = chunk_transcript(raw_asr)
        corrected_chunks = []

        for i, chunk in enumerate(chunks, start=1):
            log.info(f"  Chunk {i}/{len(chunks)}")

            query_tokens = tokenize(chunk)
            top_passages = bm25.get_top_n(query_tokens, wiki_passages, n=BM25_TOP_N)
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

        # sanity ratio check
        input_wc = len(raw_asr.split())
        output_wc = len(final_output.split())
        ratio = (output_wc / input_wc) if input_wc else 0.0
        log.info(f"  Word ratio: {ratio:.2f}")

        if ratio < 0.5:
            entry["c3_bm25_rag"] = None
            entry["c3_flag"] = "too_short"
        else:
            entry["c3_bm25_rag"] = final_output
            entry["c3_flag"] = "success"

        # incremental save
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    log.info("BM25 RAG run complete. Results saved to %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()