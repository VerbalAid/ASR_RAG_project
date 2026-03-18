from typing import List

from .rag_corpus import dense_retrieve
from .utils import MISTRAL_MODEL, chunk_with_overlap, clean_response, safe_ollama_chat


def correct_dense_generic(noisy: str,
                          passages: List[str],
                          index,
                          model) -> str:
    """
    C4a: Dense RAG over a generic/auxiliary corpus with Mistral.
    """
    chunks = chunk_with_overlap(noisy)
    out_chunks: List[str] = []
    for ch in chunks:
        ctx = dense_retrieve(ch, passages, index, model)
        context = "\n---\n".join(ctx)
        prompt = (
            "You will receive a noisy clinical conversation excerpt and some background text.\n"
            "Use the background to help with names and terminology.\n"
            "Fix clear ASR errors and output ONLY the corrected transcript text.\n\n"
            f"BACKGROUND:\n{context}\n\n"
            f"TRANSCRIPT:\n{ch}"
        )
        raw = safe_ollama_chat(MISTRAL_MODEL, prompt, num_predict=800)
        out_chunks.append(clean_response(raw))
    return " ".join(out_chunks).strip()

