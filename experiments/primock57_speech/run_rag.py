"""Run RAG correction conditions on PriMock57 speech (C2a, C2b, C3*, C4*)."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from experiments.mts.rag_corpus import (
    build_bm25_from_passages,
    build_dense_corpus,
    build_dense_index,
    dense_retrieve,
    get_dense_encoder,
    BM25_TOP_N,
    DENSE_TOP_K,
)
from experiments.mts.utils import (
    LLAMA_MODEL,
    MISTRAL_MODEL,
    chunk_with_overlap,
    clean_response,
    safe_ollama_chat,
)

OUT_DIR = ROOT / "results" / "primock57_speech"
INPUT_C1 = OUT_DIR / "c1_whisper_raw.json"
PRIMOCK57_PASSAGES = ROOT / "results" / "primock57" / "primock57_passages.json"
WIKI_CACHE = ROOT / "results" / "wiki" / "c3_lex_gen_wikipedia_passages_cache.json"
WIKI_PASSAGES_TARGET = 50000
WIKI_PASSAGE_WORD_MIN = 30
WIKI_PASSAGE_WORD_TARGET = 200
MTS_DATASET_ID = "har1/MTS_Dialogue-Clinical_Note"
MTS_PASSAGE_WORDS = 200
MTS_PASSAGE_MIN_WORDS = 30


def tokenize(text: str) -> list:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def load_wikipedia_passages() -> list:
    """Load or build Wikipedia passages (reuse TED cache if present)."""
    if WIKI_CACHE.exists():
        print(f"Loading cached Wikipedia passages from {WIKI_CACHE}...")
        with open(WIKI_CACHE, encoding="utf-8") as f:
            return json.load(f)
    print("Building Wikipedia passages (streaming)...")
    WIKI_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    passages = []
    for item in ds:
        text = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
        if not text:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        passage = " ".join(sentences[:3]).strip()
        words = passage.split()
        if len(words) < WIKI_PASSAGE_WORD_MIN:
            continue
        for i in range(0, len(words), WIKI_PASSAGE_WORD_TARGET):
            chunk = words[i : i + WIKI_PASSAGE_WORD_TARGET]
            if len(chunk) < WIKI_PASSAGE_WORD_MIN:
                continue
            passages.append(" ".join(chunk))
            if len(passages) >= WIKI_PASSAGES_TARGET:
                break
        if len(passages) >= WIKI_PASSAGES_TARGET:
            break
    print(f"Built {len(passages)} Wikipedia passages; caching to {WIKI_CACHE}")
    with open(WIKI_CACHE, "w", encoding="utf-8") as f:
        json.dump(passages, f, indent=2)
    return passages


def load_mts_dialogue_passages() -> list:
    """MTS-Dialog clean transcriptions chunked into passages (clinical corpus for C4-Den-Mat)."""
    print("Loading MTS-Dialog (clean transcriptions) for C4-Den-Mat clinical corpus...")
    ds = load_dataset(MTS_DATASET_ID)["train"]
    passages = []
    for i in range(len(ds)):
        dialogue = (ds[i].get("dialogue") or "").strip()
        if not dialogue:
            continue
        words = dialogue.split()
        for start in range(0, len(words), MTS_PASSAGE_WORDS):
            chunk = words[start : start + MTS_PASSAGE_WORDS]
            if len(chunk) < MTS_PASSAGE_MIN_WORDS:
                continue
            passages.append(" ".join(chunk))
    print(f"MTS-Dialog clinical corpus: {len(passages)} passages")
    return passages


def run_c2a(samples: list) -> None:
    """Llama-only correction (chunked)."""
    print("Running C2a (Llama only)...")
    for i, s in enumerate(samples):
        raw = s.get("c1_whisper_tiny") or ""
        if not raw.strip():
            s["c2a_llama"] = ""
            continue
        chunks = chunk_with_overlap(raw)
        out = []
        for ch in chunks:
            prompt = (
                "Fix only clear ASR errors in this transcript excerpt. "
                "Output ONLY the corrected text. No commentary.\n\n" + ch
            )
            out.append(clean_response(safe_ollama_chat(LLAMA_MODEL, prompt, num_predict=800)))
        s["c2a_llama"] = " ".join(out).strip()
    with open(OUT_DIR / "c2a_outputs.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print("  Saved c2a_outputs.json")


def run_c2b(samples: list) -> None:
    """Mistral-only correction (chunked)."""
    print("Running C2b (Mistral only)...")
    for i, s in enumerate(samples):
        raw = s.get("c1_whisper_tiny") or ""
        if not raw.strip():
            s["c2b_mistral"] = ""
            continue
        chunks = chunk_with_overlap(raw)
        out = []
        for ch in chunks:
            prompt = (
                "Fix only clear ASR errors in this transcript excerpt. "
                "Output ONLY the corrected text. No commentary.\n\n" + ch
            )
            out.append(clean_response(safe_ollama_chat(MISTRAL_MODEL, prompt, num_predict=800)))
        s["c2b_mistral"] = " ".join(out).strip()
    with open(OUT_DIR / "c2b_outputs.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print("  Saved c2b_outputs.json")


def run_c3_lex_gen(samples: list) -> None:
    """BM25 over Wikipedia (generic corpus)."""
    print("Running C3-Lex-Gen (BM25 Wikipedia)...")
    wiki_passages = load_wikipedia_passages()
    passages, bm25 = build_bm25_from_passages(wiki_passages)
    for s in samples:
        raw = s.get("c1_whisper_tiny") or ""
        if not raw.strip():
            s["c3_lex_gen"] = ""
            continue
        chunks = chunk_with_overlap(raw)
        out = []
        for ch in chunks:
            toks = tokenize(ch)
            top = bm25.get_top_n(toks, passages, n=BM25_TOP_N)
            ctx = "\n---\n".join(top)
            prompt = (
                "Use the background to help with terminology. Fix ASR errors. Output ONLY corrected text.\n\n"
                f"BACKGROUND:\n{ctx}\n\nTRANSCRIPT:\n{ch}"
            )
            out.append(clean_response(safe_ollama_chat(MISTRAL_MODEL, prompt, num_predict=800)))
        s["c3_lex_gen"] = " ".join(out).strip()
    with open(OUT_DIR / "c3_lex_gen_outputs.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print("  Saved c3_lex_gen_outputs.json")


def run_c3_lex_rel(samples: list, mts_passages: list) -> None:
    """BM25 over MTS-Dialog clinical transcriptions (same corpus as C4-Den-Mat)."""
    print("Running C3-Lex-Rel (BM25 MTS-Dialog clinical)...")
    if not mts_passages:
        print("  No MTS passages; skipping.")
        for s in samples:
            s["c3_lex_rel"] = ""
        with open(OUT_DIR / "c3_lex_rel_outputs.json", "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
        return
    passages, bm25 = build_bm25_from_passages(mts_passages)
    for s in samples:
        raw = s.get("c1_whisper_tiny") or ""
        if not raw.strip():
            s["c3_lex_rel"] = ""
            continue
        chunks = chunk_with_overlap(raw)
        out = []
        for ch in chunks:
            toks = tokenize(ch)
            top = bm25.get_top_n(toks, passages, n=BM25_TOP_N)
            ctx = "\n---\n".join(top)
            prompt = (
                "Use the background to help with terminology. Fix ASR errors. Output ONLY corrected text.\n\n"
                f"BACKGROUND:\n{ctx}\n\nTRANSCRIPT:\n{ch}"
            )
            out.append(clean_response(safe_ollama_chat(MISTRAL_MODEL, prompt, num_predict=800)))
        s["c3_lex_rel"] = " ".join(out).strip()
    with open(OUT_DIR / "c3_lex_rel_outputs.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print("  Saved c3_lex_rel_outputs.json")


def run_c3_lex_mat(samples: list, primock_by_id: dict) -> None:
    """BM25 leave-one-out (exclude current consultation's passages)."""
    print("Running C3-Lex-Mat (BM25 LOO)...")
    for s in samples:
        cid = s.get("speaker_id", "")
        passages_loo = [p for sid, plist in primock_by_id.items() if sid != cid for p in plist if len(p.split()) >= 30]
        if not passages_loo:
            s["c3_lex_mat"] = ""
            continue
        _, bm25 = build_bm25_from_passages(passages_loo)
        raw = s.get("c1_whisper_tiny") or ""
        if not raw.strip():
            s["c3_lex_mat"] = ""
            continue
        chunks = chunk_with_overlap(raw)
        out = []
        for ch in chunks:
            toks = tokenize(ch)
            top = bm25.get_top_n(toks, passages_loo, n=BM25_TOP_N)
            ctx = "\n---\n".join(top)
            prompt = (
                "Use the background to help with terminology. Fix ASR errors. Output ONLY corrected text.\n\n"
                f"BACKGROUND:\n{ctx}\n\nTRANSCRIPT:\n{ch}"
            )
            out.append(clean_response(safe_ollama_chat(MISTRAL_MODEL, prompt, num_predict=800)))
        s["c3_lex_mat"] = " ".join(out).strip()
    with open(OUT_DIR / "c3_lex_mat_outputs.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print("  Saved c3_lex_mat_outputs.json")


def run_c4_den_gen(samples: list) -> None:
    """Dense retrieval over Wikipedia (generic corpus)."""
    print("Running C4-Den-Gen (dense Wikipedia)...")
    wiki_passages = load_wikipedia_passages()
    passages, index, model = build_dense_corpus(wiki_passages)
    for s in samples:
        raw = s.get("c1_whisper_tiny") or ""
        if not raw.strip():
            s["c4_den_gen"] = ""
            continue
        chunks = chunk_with_overlap(raw)
        out = []
        for ch in chunks:
            ctx = dense_retrieve(ch, passages, index, model, top_k=DENSE_TOP_K)
            prompt = (
                "Use the background to help with terminology. Fix ASR errors. Output ONLY corrected text.\n\n"
                f"BACKGROUND:\n" + "\n---\n".join(ctx) + f"\n\nTRANSCRIPT:\n{ch}"
            )
            out.append(clean_response(safe_ollama_chat(MISTRAL_MODEL, prompt, num_predict=800)))
        s["c4_den_gen"] = " ".join(out).strip()
    with open(OUT_DIR / "c4_den_gen_outputs.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print("  Saved c4_den_gen_outputs.json")


def run_c4_den_mat(samples: list, primock_by_id: dict) -> None:
    """Dense retrieval over leave-one-out PriMock57 (same corpus as C3-Lex-Mat)."""
    print("Running C4-Den-Mat (dense LOO PriMock57)...")
    model = get_dense_encoder()
    for s in samples:
        cid = s.get("speaker_id", "")
        passages_loo = [p for sid, plist in primock_by_id.items() if sid != cid for p in plist if len(p.split()) >= 30]
        if not passages_loo:
            s["c4_den_mat"] = ""
            continue
        _, index = build_dense_index(passages_loo, model)
        raw = s.get("c1_whisper_tiny") or ""
        if not raw.strip():
            s["c4_den_mat"] = ""
            continue
        chunks = chunk_with_overlap(raw)
        out = []
        for ch in chunks:
            ctx = dense_retrieve(ch, passages_loo, index, model, top_k=DENSE_TOP_K)
            prompt = (
                "Use the background to help with terminology. Fix ASR errors. Output ONLY corrected text.\n\n"
                f"BACKGROUND:\n" + "\n---\n".join(ctx) + f"\n\nTRANSCRIPT:\n{ch}"
            )
            out.append(clean_response(safe_ollama_chat(MISTRAL_MODEL, prompt, num_predict=800)))
        s["c4_den_mat"] = " ".join(out).strip()
    with open(OUT_DIR / "c4_den_mat_outputs.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print("  Saved c4_den_mat_outputs.json")


def main():
    parser = argparse.ArgumentParser(description="Run PriMock57 speech RAG conditions.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["c2a", "c2b", "c3_lex_gen", "c3_lex_rel", "c3_lex_mat", "c4_den_gen", "c4_den_mat"],
        default=None,
        help="Run only these conditions (default: all)",
    )
    args = parser.parse_args()
    only = set(args.only) if args.only else None

    if not INPUT_C1.exists():
        print(f"Error: Run c1_whisper.py first. Missing {INPUT_C1}", file=sys.stderr)
        sys.exit(1)
    if not PRIMOCK57_PASSAGES.exists():
        print(f"Error: Run analysis/prepare_primock57.py first. Missing {PRIMOCK57_PASSAGES}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_C1, encoding="utf-8") as f:
        samples = json.load(f)
    with open(PRIMOCK57_PASSAGES, encoding="utf-8") as f:
        primock_list = json.load(f)
    primock_by_id = {}
    for p in primock_list:
        sid = p.get("source_consultation_id", "")
        pas = (p.get("passage") or "").strip()
        if sid and pas:
            primock_by_id.setdefault(sid, []).append(pas)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def run_if(cond: str, fn):
        if only is None or cond in only:
            fn()

    run_if("c2a", lambda: run_c2a(samples))
    with open(INPUT_C1, encoding="utf-8") as f:
        samples = json.load(f)
    run_if("c2b", lambda: run_c2b(samples))
    with open(INPUT_C1, encoding="utf-8") as f:
        samples = json.load(f)
    run_if("c3_lex_gen", lambda: run_c3_lex_gen(samples))
    with open(INPUT_C1, encoding="utf-8") as f:
        samples = json.load(f)
    mts_passages = load_mts_dialogue_passages() if (only is None or "c3_lex_rel" in only) else []
    run_if("c3_lex_rel", lambda: run_c3_lex_rel(samples, mts_passages))
    with open(INPUT_C1, encoding="utf-8") as f:
        samples = json.load(f)
    run_if("c3_lex_mat", lambda: run_c3_lex_mat(samples, primock_by_id))
    with open(INPUT_C1, encoding="utf-8") as f:
        samples = json.load(f)
    run_if("c4_den_gen", lambda: run_c4_den_gen(samples))
    with open(INPUT_C1, encoding="utf-8") as f:
        samples = json.load(f)
    run_if("c4_den_mat", lambda: run_c4_den_mat(samples, primock_by_id))

    print("Done. All condition outputs in results/primock57_speech/")


if __name__ == "__main__":
    main()
