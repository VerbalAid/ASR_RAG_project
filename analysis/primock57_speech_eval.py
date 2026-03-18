"""Compute WER, BLEU, ROUGE-L, BERTScore per sample/condition for PriMock57 speech."""
import json
import re
from pathlib import Path

import numpy as np
from bert_score import score as bert_score
from jiwer import wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "primock57_speech"
METRICS_DIR = RESULTS_DIR / "metrics"

CONDITION_FILES = {
    "c1": ("c1_whisper_raw.json", "c1_whisper_tiny"),
    "c2a": ("c2a_outputs.json", "c2a_llama"),
    "c2b": ("c2b_outputs.json", "c2b_mistral"),
    "c3_lex_gen": ("c3_lex_gen_outputs.json", "c3_lex_gen"),
    "c3_lex_rel": ("c3_lex_rel_outputs.json", "c3_lex_rel"),
    "c3_lex_mat": ("c3_lex_mat_outputs.json", "c3_lex_mat"),
    "c4_den_gen": ("c4_den_gen_outputs.json", "c4_den_gen"),
    "c4_den_mat": ("c4_den_mat_outputs.json", "c4_den_mat"),
}


def normalise(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_metrics(ref: str, hyp: str, scorer, smoothing) -> dict:
    ref_n = normalise(ref)
    hyp_n = normalise(hyp)
    if not hyp_n:
        return {"wer": 1.0, "bleu": 0.0, "rouge_l": 0.0, "bert_score": 0.0}
    w = wer(ref_n, hyp_n)
    b = sentence_bleu([ref_n.split()], hyp_n.split(), smoothing_function=smoothing)
    r = scorer.score(ref_n, hyp_n)["rougeL"].fmeasure
    _, _, f1 = bert_score([hyp_n], [ref_n], lang="en", verbose=False)
    bert = float(f1.item())
    return {"wer": w, "bleu": b, "rouge_l": r, "bert_score": bert}


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    # Load C1 for ground truth and speaker list
    c1_path = RESULTS_DIR / "c1_whisper_raw.json"
    if not c1_path.exists():
        print(f"Error: Missing {c1_path}. Run PriMock57 speech C1 first.")
        return
    with c1_path.open(encoding="utf-8") as f:
        c1_data = json.load(f)

    for cond, (filename, hyp_key) in CONDITION_FILES.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"Skipping {cond}: {path} not found.")
            continue
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for sample in data:
            ref = sample.get("ground_truth", "")
            hyp = sample.get(hyp_key, "")
            if not ref:
                continue
            m = compute_metrics(ref, hyp, scorer, smoothing)
            rows.append({
                "sample_id": sample.get("sample_id"),
                "speaker_id": sample.get("speaker_id"),
                "wer": round(m["wer"], 4),
                "bleu": round(m["bleu"], 4),
                "rouge_l": round(m["rouge_l"], 4),
                "bert_score": round(m["bert_score"], 4),
            })
            print(f"  {sample.get('speaker_id', '')}: WER={m['wer']:.3f} BLEU={m['bleu']:.3f}")

        if rows:
            out_path = METRICS_DIR / f"{cond}_metrics.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
            print(f"Saved {out_path} (n={len(rows)})")
            avg_wer = np.mean([r["wer"] for r in rows])
            print(f"  Mean WER: {avg_wer:.4f}\n")

    print("PriMock57 speech eval done.")


if __name__ == "__main__":
    main()
