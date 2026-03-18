import json
import re
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ── 1. Load TED transcripts as corpus ─────────────────────────────────────
with open('results/ted/c1_whisper_raw.json', 'r') as f:
    results = json.load(f)

def chunk_text(text, max_words=400):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# ── 2. Build per-speaker corpus (exclude test speaker to avoid leakage) ───
print("Building TED corpus...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index_for_speaker(test_speaker):
    """Build a FAISS index from all TED transcripts EXCEPT the test speaker."""
    passages = []
    for r in results:
        if r['speaker_id'] == test_speaker:
            continue  # exclude to prevent leakage
        chunks = chunk_text(r['ground_truth'], max_words=150)
        passages.extend(chunks)

    embeddings = model.encode(passages, batch_size=256,
                               show_progress_bar=False,
                               convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, passages

def retrieve_context(chunk, index, passages, top_k=3):
    embedding = model.encode([chunk], convert_to_numpy=True)
    distances, indices = index.search(embedding, top_k)
    return [passages[idx] for idx in indices[0]]

def correct_with_ted_rag(text, test_speaker):
    index, passages = build_index_for_speaker(test_speaker)
    chunks = chunk_text(text, max_words=400)
    corrected_chunks = []

    for chunk in chunks:
        context_passages = retrieve_context(chunk, index, passages, top_k=3)
        context_str = "\n---\n".join(context_passages)

        prompt = (
            f"You have the following relevant context from similar TED talks:\n\n"
            f"{context_str}\n\n"
            f"---\n"
            f"Using the context above to help identify correct names, terms, and phrases, "
            f"fix only clear ASR transcription errors in the text below. "
            f"Output ONLY the corrected text. No commentary, no notes, no preamble.\n\n"
            f"{chunk}"
        )

        response = ollama.chat(
            model="mistral:latest",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"].strip()
        raw = re.sub(r"^(here is|corrected transcript|note:).{0,100}\n",
                     "", raw, flags=re.IGNORECASE)
        corrected_chunks.append(raw.strip())

    return " ".join(corrected_chunks)

# ── 3. Run C4b (TED-domain dense RAG) over all speakers ───────────────────
print("Running C4b RAG correction (TED corpus)...")
for i, result in enumerate(results):
    print(f"Processing {i+1}/{len(results)}: {result['speaker_id']}...")
    corrected = correct_with_ted_rag(
        result['c1_whisper_tiny'],
        result['speaker_id']
    )
    input_len  = len(result['c1_whisper_tiny'].split())
    output_len = len(corrected.split())
    ratio      = output_len / input_len
    flag       = "⚠️ SUSPECT" if ratio < 0.7 else "✅ OK"
    print(f"  {flag} {input_len}w → {output_len}w (ratio: {ratio:.2f})")
    result['c4b_rag_ted'] = corrected

with open('results/ted/c4_den_mat_outputs.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nDone. Saved to results/ted/c4_den_mat_outputs.json")