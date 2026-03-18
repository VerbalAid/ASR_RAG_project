import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List

from datasets import load_dataset

from . import noise
from .mts_c3_bm25 import correct_bm25
from .mts_c4_dense_clinical import correct_dense_clinical
from .mts_c4_dense_generic import correct_dense_generic
from .mts_c2_llama import correct_llama
from .mistral import correct_mistral
from .rag_corpus import (
    build_bm25_corpus_from_ag_news,
    build_bm25_corpus_from_primock57,
    build_bm25_from_passages,
    build_dense_corpus,
    build_dense_index,
    get_dense_encoder,
)
from .utils import RESULTS_DIR


MTS_DATASET_ID = "har1/MTS_Dialogue-Clinical_Note"
MAX_DIALOGUES = 100
C4A_CORPUS_CHOICES = {"ag_news", "primock57"}
C3_CORPUS_CHOICES = {"ag_news", "primock57"}


def load_subset(max_dialogues: int = MAX_DIALOGUES):
    ds = load_dataset(MTS_DATASET_ID)["train"]
    indices = list(range(len(ds)))
    random.seed(42)
    random.shuffle(indices)
    return ds, indices[:max_dialogues]


def load_primock57_passages() -> List[str]:
    path = Path("results") / "primock57" / "primock57_passages.json"
    if not path.exists():
        raise FileNotFoundError(
            f"PriMock57 passages not found at {path}. Run analysis/prepare_primock57.py first."
        )
    with path.open("r") as f:
        data = json.load(f)
    passages = [row.get("passage", "").strip() for row in data if row.get("passage", "").strip()]
    if not passages:
        raise RuntimeError(f"No valid passages found in {path}")
    return passages


def run_condition(
    cond: str,
    max_dialogues: int = MAX_DIALOGUES,
    *,
    c4a_corpus: str = "ag_news",
    c3_corpus: str = "ag_news",
    resume: bool = False,
) -> Path:
    """
    Run a single condition over a subset of MTS-Dialog and write results JSON.

    cond in {"c1", "c2a", "c2b", "c3", "c3_mat", "c4a", "c4b"}.
    When cond is "c3" and c3_corpus is "primock57", uses PriMock57 (writes mts_c3b_results.json); ag_news writes mts_c3a_results.json.
    If resume=True and the output file already exists, skips dialogues already present and appends only new ones.
    """
    ds, idxs = load_subset(max_dialogues)

    if cond == "c3":
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / ("mts_c3_lex_rel_results.json" if c3_corpus == "primock57" else "mts_c3_lex_gen_results.json")
    elif cond == "c4a" and c4a_corpus == "primock57":
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "mts_c4_den_rel_results.json"
    elif cond == "c4a" and c4a_corpus == "ag_news":
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "mts_c4_den_gen_results.json"
    elif cond == "c4b":
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "mts_c4_den_mat_results.json"
    elif cond == "c3_mat":
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "mts_c3_lex_mat_results.json"
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"mts_{cond}_results.json"

    existing_by_id: Dict[int, Dict[str, Any]] = {}
    if resume and out_path.exists():
        with out_path.open("r") as f:
            existing_list = json.load(f)
        existing_by_id = {int(r["sample_id"]): r for r in existing_list}
        print(f"[resume] Loaded {len(existing_by_id)} existing records from {out_path}")

    out_records: List[Dict[str, Any]] = []

    bm25_passages = bm25_index = None
    dense_gen_passages = dense_gen_index = dense_gen_model = None
    dense_clin_passages = dense_clin_index = dense_clin_model = None

    if cond == "c3":
        if c3_corpus == "primock57":
            primock_passages = load_primock57_passages()
            bm25_passages, bm25_index = build_bm25_corpus_from_primock57(primock_passages)
        else:
            bm25_passages, bm25_index = build_bm25_corpus_from_ag_news()
    if cond in {"c4a", "c4b"}:
        if cond == "c4a" and c4a_corpus == "primock57":
            dense_source_passages = load_primock57_passages()
            print(f"Using PriMock57 dense corpus for C4a_domain: {len(dense_source_passages)} passages")
        elif cond == "c4a" and c4a_corpus == "ag_news":
            ag_passages, _ = build_bm25_corpus_from_ag_news()
            dense_source_passages = ag_passages
            print(f"Using AG News dense corpus for C4a (generic): {len(dense_source_passages)} passages")
        else:
            dense_source_passages = None
        if dense_source_passages is not None:
            dense_gen_passages, dense_gen_index, dense_gen_model = build_dense_corpus(dense_source_passages)
    if cond == "c4b":
        dense_clin_model = get_dense_encoder()
        print("C4b: using leave-one-out over clinical notes (no data leakage)")

    for rank, i in enumerate(idxs, start=1):
        row = ds[i]
        dialogue = row["dialogue"]
        sample_id = int(row["ID"])
        clean = (dialogue or "").strip()
        if not clean:
            continue

        if sample_id in existing_by_id:
            out_records.append(existing_by_id[sample_id])
            print(f"\n=== Sample {rank}/{len(idxs)} | ID={sample_id} | cond={cond} [resumed] ===")
            continue

        print(f"\n=== Sample {rank}/{len(idxs)} | ID={sample_id} | cond={cond} ===")
        noisy = noise.simulate_asr_noise(clean)

        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "dialogue": clean,
            "noisy": noisy,
        }

        if cond == "c1":
            pass
        elif cond == "c2a":
            record["c2a_llama"] = correct_llama(noisy)
        elif cond == "c2b":
            record["c2b_mistral"] = correct_mistral(noisy)
        elif cond == "c3":
            assert bm25_passages is not None and bm25_index is not None
            record["c3_bm25_rag"] = correct_bm25(noisy, bm25_passages, bm25_index)
        elif cond == "c3_mat":
            other_notes = [ds[j]["section_text"] for j in idxs if j != i and (ds[j].get("section_text") or "").strip()]
            if not other_notes:
                record["c3_bm25_rag"] = correct_mistral(noisy)
            else:
                mat_passages, mat_bm25 = build_bm25_from_passages(other_notes)
                record["c3_bm25_rag"] = correct_bm25(noisy, mat_passages, mat_bm25)
        elif cond == "c4a":
            assert dense_gen_passages is not None and dense_gen_index is not None and dense_gen_model is not None
            record["c4a_dense_generic"] = correct_dense_generic(
                noisy, dense_gen_passages, dense_gen_index, dense_gen_model
            )
        elif cond == "c4b":
            other_notes = [ds[j]["section_text"] for j in idxs if j != i and (ds[j].get("section_text") or "").strip()]
            if not other_notes:
                record["c4b_dense_clinical"] = correct_mistral(noisy)
            else:
                mat_passages, mat_index = build_dense_index(other_notes, dense_clin_model)
                record["c4b_dense_clinical"] = correct_dense_clinical(
                    noisy, mat_passages, mat_index, dense_clin_model
                )
        else:
            raise ValueError(f"Unknown condition: {cond}")

        out_records.append(record)

        with out_path.open("w") as f:
            json.dump(out_records, f, indent=2)
        if rank % 10 == 0 or rank == len(idxs):
            print(f"  [saved {len(out_records)}/{len(idxs)}]")

    with out_path.open("w") as f:
        json.dump(out_records, f, indent=2)
    print(f"\nSaved {cond} results to {out_path} ({len(out_records)} records)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run a single MTS-Dialog clinical condition.")
    parser.add_argument(
        "condition",
        choices=["c1", "c2a", "c2b", "c3", "c3_mat", "c4a", "c4b"],
        help="Which condition to run (c1, c2a, c2b, c3, c3_mat, c4a, c4b).",
    )
    parser.add_argument(
        "--max-dialogues",
        type=int,
        default=MAX_DIALOGUES,
        help=f"Number of dialogues to sample (default: {MAX_DIALOGUES}).",
    )
    parser.add_argument(
        "--c4a-corpus",
        choices=sorted(C4A_CORPUS_CHOICES),
        default="ag_news",
        help="Dense corpus for c4a only: ag_news (default) or primock57.",
    )
    parser.add_argument(
        "--c3-corpus",
        choices=sorted(C3_CORPUS_CHOICES),
        default="ag_news",
        help="BM25 corpus for c3: ag_news (C3-Lex-Gen) or primock57 (C3-Lex-Rel).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume: load existing output and skip dialogues already present (saves after each new record).",
    )
    args = parser.parse_args()
    run_condition(
        args.condition,
        args.max_dialogues,
        c4a_corpus=args.c4a_corpus,
        c3_corpus=args.c3_corpus,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

