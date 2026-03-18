"""
MTS-Dialog clinical pipeline package.

This package exposes separate modules for:
- noise       (C1 noisy transcripts)
- llama       (C2a LLaMA-only correction)
- mistral     (C2b Mistral-only correction)
- bm25        (C3 BM25 lexical RAG)
- dense_generic  (C4a dense RAG with generic corpus)
- dense_clinical (C4b dense RAG with clinical notes)
- rag_corpus  (shared BM25/dense corpus builders)
- utils       (chunking, Ollama calls, text cleaning)
- run         (entrypoint to run a single condition)
"""

