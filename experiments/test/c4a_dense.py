import json
import re
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset

# ── 1. Load a generic Wikipedia corpus for retrieval ──────────────────────
print("Loading Wikipedia corpus...")
wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

# Grab ~10,000 passages (first 2-3 sentences of each article) to avoid OOM
passages = []
for i, article in enumerate(wiki):
    text = article["text"]
    # Take first 3 sentences as a passage
    sentences = text.split(". ")
    passage = ". ".join(sentences[:3]).strip()
    if len(passage.split()) > 20:  # skip very short ones
        passages.append(passage)
    if len(passages) >= 10000:
        break
    if i % 5000 == 0:
        print(f"  Collected {len(passages)} passages...")

print(f"Corpus ready: {len(passages)} passages")

# ── 2. Embed the corpus and build FAISS index ─────────────────────────────
print("\nEmbedding corpus on CPU (this takes a few minutes)...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
corpus_embeddings = model.encode(
    passages,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)
print(f"FAISS index built: {index.ntotal} vectors, dimension {dimension}")

# ── 3. Load C1 transcripts ────────────────────────────────────────────────
with open("results/ted/c1_whisper_raw.json", "r") as f:
    results = json.load(f)

# ── 4. Helper functions ───────────────────────────────────────────────────
def chunk_text(text, max_words=400):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def retrieve_context(chunk, top_k=3):
    """Embed a chunk and retrieve top_k relevant Wikipedia passages."""
    embedding = model.encode([chunk], convert_to_numpy=True)
    distances, indices = index.search(embedding, top_k)
    return [passages[idx] for idx in indices[0]]

def correct_with_rag(text):
    chunks = chunk_text(text, max_words=400)
    corrected_chunks = []

    for chunk in chunks:
        # Retrieve relevant context for this chunk
        context_passages = retrieve_context(chunk, top_k=3)
        context_str = "\n---\n".join(context_passages)

        prompt = (
            f"You have the following relevant background context:\n\n"
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
        # Strip any meta-commentary that slips through
        raw = re.sub(r"^(here is|corrected transcript|note:).{0,100}\n",
                     "", raw, flags=re.IGNORECASE)
        corrected_chunks.append(raw.strip())

    return " ".join(corrected_chunks)

# ── 5. Run C3 over all speakers ───────────────────────────────────────────
print("\nRunning C3 RAG correction...")
for i, result in enumerate(results):
    print(f"Processing {i+1}/{len(results)}: {result['speaker_id']}...")
    corrected = correct_with_rag(result["c1_whisper_tiny"])

    input_len  = len(result["c1_whisper_tiny"].split())
    output_len = len(corrected.split())
    ratio      = output_len / input_len
    flag       = "⚠️ SUSPECT" if ratio < 0.7 else "✅ OK"
    print(f"  {flag} {input_len}w → {output_len}w (ratio: {ratio:.2f})")

    result["c3_rag_generic"] = corrected

with open("results/ted/c3_outputs.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Saved to results/ted/c3_outputs.json")