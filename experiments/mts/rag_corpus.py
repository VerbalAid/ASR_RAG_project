import re
from typing import List, Tuple

import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


BM25_PASSAGE_WORD_MIN = 30
BM25_PASSAGE_WORD_TARGET = 200
BM25_MAX_PASSAGES = 50000
BM25_TOP_N = 3
DENSE_TOP_K = 3


def _pick_torch_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def build_bm25_corpus_from_ag_news():
    """
    Build a simple BM25 corpus from AG News text.
    Returns (passages, bm25_index).
    """
    from rank_bm25 import BM25Okapi

    print("Building BM25 corpus from AG News (fallback)...")
    ds = load_dataset("ag_news", split="train")
    passages: List[str] = []
    for item in ds:
        text = item.get("text", "")
        if not text:
            continue
        words = text.split()
        if len(words) < BM25_PASSAGE_WORD_MIN:
            continue
        for i in range(0, len(words), BM25_PASSAGE_WORD_TARGET):
            chunk = words[i : i + BM25_PASSAGE_WORD_TARGET]
            if len(chunk) < BM25_PASSAGE_WORD_MIN:
                continue
            passages.append(" ".join(chunk))
            if len(passages) >= BM25_MAX_PASSAGES:
                break
        if len(passages) >= BM25_MAX_PASSAGES:
            break

    print(f"BM25 corpus size: {len(passages)}")
    tokenized = [re.sub(r"[^\w\s]", "", p.lower()).split() for p in passages]
    bm25 = BM25Okapi(tokenized)
    return passages, bm25


def build_bm25_from_passages(passages: List[str], min_words: int = BM25_PASSAGE_WORD_MIN):
    """
    Build BM25 index from an arbitrary list of text passages.
    Returns (passages, bm25_index). Used for C3-Lex-Mat (leave-one-out clinical notes).
    """
    from rank_bm25 import BM25Okapi

    passages = [p.strip() for p in passages if p and len(p.strip().split()) >= min_words]
    if not passages:
        raise RuntimeError("No valid passages for BM25.")
    tokenized = [re.sub(r"[^\w\s]", "", p.lower()).split() for p in passages]
    bm25 = BM25Okapi(tokenized)
    return passages, bm25


def build_bm25_corpus_from_primock57(primock57_passages: List[str]):
    """
    Build BM25 corpus from PriMock57 passages (same 67 passages as C4a).
    Returns (passages, bm25_index). Use for C3_fixed to match C4a corpus.
    """
    print(f"Building BM25 corpus from PriMock57 ({len(primock57_passages)} passages)...")
    return build_bm25_from_passages(primock57_passages)


def get_dense_encoder() -> SentenceTransformer:
    """Load the dense encoder model once (for leave-one-out reuse)."""
    device = _pick_torch_device()
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)


def build_dense_index(passages: List[str], model: SentenceTransformer) -> Tuple[List[str], faiss.IndexFlatL2]:
    """
    Build FAISS index from passages using an existing model.
    Returns (passages, index). Use for leave-one-out when model is reused.
    """
    if not passages:
        raise RuntimeError("No passages for dense index.")
    embeddings = model.encode(
        passages,
        batch_size=256,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return passages, index


def build_dense_corpus(texts: List[str]) -> Tuple[List[str], faiss.IndexFlatL2, SentenceTransformer]:
    """
    Build a dense corpus (SentenceTransformer + FAISS IndexFlatL2).
    Returns (passages, index, model).
    """
    print(f"Building dense corpus with {len(texts)} passages...")
    model = get_dense_encoder()
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index: {index.ntotal} vectors, dim={dim}")
    return texts, index, model


def dense_retrieve(query: str,
                   passages: List[str],
                   index: faiss.IndexFlatL2,
                   model: SentenceTransformer,
                   top_k: int = DENSE_TOP_K) -> List[str]:
    """Retrieve top_k dense nearest neighbours for a query string."""
    embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(embedding, top_k)
    return [passages[i] for i in indices[0]]

