import re

from .rag_corpus import BM25_TOP_N
from .utils import MISTRAL_MODEL, clean_response, safe_ollama_chat


def correct_bm25(noisy: str, bm25_passages, bm25_index) -> str:
    """
    C3: BM25 lexical RAG over AG News with Mistral correction.
    """
    tokens = re.sub(r"[^\w\s]", "", noisy.lower()).split()
    top = bm25_index.get_top_n(tokens, bm25_passages, n=BM25_TOP_N)
    context = "\n\n".join(top)
    prompt = (
        "You will receive a noisy clinical conversation excerpt and some background text.\n"
        "Use the background only to help with named entities and terminology.\n"
        "Fix clear ASR errors and output ONLY the corrected text.\n\n"
        f"BACKGROUND:\n{context}\n\n"
        f"TRANSCRIPT:\n{noisy}"
    )
    raw = safe_ollama_chat(MISTRAL_MODEL, prompt)
    return clean_response(raw)

