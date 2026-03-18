import re
import time
from pathlib import Path
from typing import List, Dict, Any

import ollama


# ========= CONFIG =========
# All MTS-Dialog outputs/metrics live under this subfolder.
RESULTS_DIR = Path("results") / "mts"

CHUNK_WORD_TARGET = 400
CHUNK_OVERLAP_WORDS = 50

LLAMA_MODEL = "llama3:8b"
MISTRAL_MODEL = "mistral:latest"  # must match C2b/C4; do not switch to another model mid-experiment

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0


def chunk_with_overlap(text: str,
                       chunk_size: int = CHUNK_WORD_TARGET,
                       overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    """Simple word-based sliding window."""
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


def safe_ollama_chat(model: str,
                     user_content: str,
                     *,
                     temperature: float = 0.0,
                     top_p: float = 1.0,
                     num_predict: int | None = None,
                     max_retries: int = MAX_RETRIES,
                     base_backoff: float = BASE_BACKOFF_SECONDS) -> str:
    """Call Ollama chat API with basic retry/backoff."""
    messages: List[Dict[str, str]] = [{"role": "user", "content": user_content}]
    options: Dict[str, Any] = {"temperature": temperature, "top_p": top_p}
    if num_predict is not None:
        options["num_predict"] = num_predict
    for attempt in range(1, max_retries + 1):
        try:
            resp: Any = ollama.chat(
                model=model,
                messages=messages,
                options=options,
            )
            if isinstance(resp, dict):
                msg = resp.get("message") or {}
                content = msg.get("content") or ""
            else:
                content = str(resp)
            if content:
                return content
        except Exception as e:
            print(f"[{model}] attempt {attempt}/{max_retries} failed: {e}")
        time.sleep(base_backoff * (2 ** (attempt - 1)))
    return ""


PREAMBLE_PATTERN = re.compile(
    r"^(here is( the)? corrected transcript:?|corrected text:?|"
    r"here's the correction:?|note:).*?\n",
    flags=re.IGNORECASE | re.DOTALL,
)


def clean_response(raw: str, preserve_newlines: bool = False) -> str:
    """
    Basic cleanup for LLM outputs.
    - Strips common preambles and code fences.
    - If preserve_newlines is False (default), collapses all whitespace.
    - If preserve_newlines is True, collapses only intra-line spaces/tabs and keeps line breaks.
    """
    if not raw:
        return ""
    text = raw.strip()
    text = PREAMBLE_PATTERN.sub("", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    if preserve_newlines:
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
    else:
        text = re.sub(r"\s+", " ", text)
    return text.strip()

