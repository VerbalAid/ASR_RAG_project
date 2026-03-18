"""Run C2b and C4-Den-Mat (LOO PriMock57) on full 57 consultations. Optional: --add-c4-den-gen."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from experiments.mts.rag_corpus import (
    build_dense_corpus,
    build_dense_index,
    dense_retrieve,
    get_dense_encoder,
    DENSE_TOP_K,
)
from experiments.mts.utils import (
    MISTRAL_MODEL,
    chunk_with_overlap,
    clean_response,
    safe_ollama_chat,
)

OUT_DIR = ROOT / "results" / "primock57_full_57"
INPUT_C1 = OUT_DIR / "c1_whisper_raw.json"
PRIMOCK57_PASSAGES = ROOT / "results" / "primock57" / "primock57_passages.json"
WIKI_CACHE = ROOT / "results" / "wiki" / "c3_lex_gen_wikipedia_passages_cache.json"
WIKI_PASSAGES_TARGET = 50000
WIKI_PASSAGE_WORD_MIN = 30
WIKI_PASSAGE_WORD_TARGET = 200


def load_wikipedia_passages() -> list:
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


def load_primock_by_id() -> dict:
    """Load PriMock57 passages and group by consultation id (for LOO)."""
    if not PRIMOCK57_PASSAGES.exists():
        raise FileNotFoundError(
            f"PriMock57 passages not found at {PRIMOCK57_PASSAGES}. Run analysis/prepare_primock57.py first."
        )
    with open(PRIMOCK57_PASSAGES, encoding="utf-8") as f:
        primock_list = json.load(f)
    primock_by_id = {}
    for p in primock_list:
        sid = p.get("source_consultation_id", "")
        pas = (p.get("passage") or "").strip()
        if sid and pas:
            primock_by_id.setdefault(sid, []).append(pas)
    print(f"Loaded PriMock57 passages for {len(primock_by_id)} consultations (for C4-Den-Mat LOO)")
    return primock_by_id


def run_c2b(samples: list) -> None:
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
        if (i + 1) % 10 == 0:
            print(f"  C2b {i+1}/{len(samples)}")
    with open(OUT_DIR / "c2b_outputs.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print("  Saved c2b_outputs.json")


def run_c4_den_mat(samples: list, primock_by_id: dict) -> None:
    """C4-Den-Mat with leave-one-out PriMock57 (same corpus as C3-Lex-Mat)."""
    print("Running C4-Den-Mat (dense LOO PriMock57)...")
    model = get_dense_encoder()
    for i, s in enumerate(samples):
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
        if (i + 1) % 10 == 0:
            print(f"  C4-Den-Mat {i+1}/{len(samples)}")
    with open(OUT_DIR / "c4_den_mat_outputs.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print("  Saved c4_den_mat_outputs.json")


def run_c4_den_gen(samples: list) -> None:
    print("Running C4-Den-Gen (dense Wikipedia)...")
    wiki_passages = load_wikipedia_passages()
    passages, index, model = build_dense_corpus(wiki_passages)
    for i, s in enumerate(samples):
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
        if (i + 1) % 10 == 0:
            print(f"  C4-Den-Gen {i+1}/{len(samples)}")
    with open(OUT_DIR / "c4_den_gen_outputs.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print("  Saved c4_den_gen_outputs.json")


def main():
    parser = argparse.ArgumentParser(description="Run C2b and C4-Den-Mat on PriMock57 full-57.")
    parser.add_argument(
        "--add-c4-den-gen",
        action="store_true",
        help="Also run C4-Den-Gen (generic dense Wikipedia).",
    )
    args = parser.parse_args()

    if not INPUT_C1.exists():
        print(f"Error: Run c1_whisper_57.py first. Missing {INPUT_C1}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_C1, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from {INPUT_C1}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_c2b(samples)

    with open(INPUT_C1, encoding="utf-8") as f:
        samples = json.load(f)
    primock_by_id = load_primock_by_id()
    run_c4_den_mat(samples, primock_by_id)

    if args.add_c4_den_gen:
        with open(INPUT_C1, encoding="utf-8") as f:
            samples = json.load(f)
        run_c4_den_gen(samples)

    print("Done. Outputs in results/primock57_full_57/")


if __name__ == "__main__":
    main()
