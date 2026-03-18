"""Evaluate TED C3-Lex-Rel outputs; write per-speaker metrics and CSV."""
import json
import re
from pathlib import Path

from jiwer import wer
from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

TED_PATH = Path("results") / "ted"
INPUT = TED_PATH / "ted_c3_lex_rel_results.json"
METRICS_OUT = TED_PATH / "metrics" / "c3_lex_rel_metrics.json"
PER_SPEAKER_CSV = TED_PATH / "per_speaker" / "ted_per_speaker_c3_lex_rel.csv"


def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    if not INPUT.exists():
        print(f"Input not found: {INPUT}")
        return

    with INPUT.open("r") as f:
        results = json.load(f)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    metrics_rows = []

    for sample in results:
        gt = normalise(sample.get("ground_truth", ""))
        if not gt:
            print(f"SKIP {sample.get('speaker_id')} — missing ground truth")
            continue

        hyp_raw = sample.get("c3_lex_rel_bm25_rag") or ""
        hyp = normalise(hyp_raw)
        if not hyp:
            print(f"SKIP {sample.get('speaker_id')} — empty C3-Lex-Rel output")
            continue

        w = wer(gt, hyp)
        b = sentence_bleu([gt.split()], hyp.split(), smoothing_function=smoothing)
        rl = scorer.score(gt, hyp)["rougeL"].fmeasure
        _, _, f1 = score([hyp], [gt], lang="en", verbose=False)
        bert = f1.mean().item()

        metrics_rows.append({
            "sample_id": sample.get("sample_id"),
            "speaker_id": sample.get("speaker_id"),
            "wer": round(w, 4),
            "bleu": round(b, 4),
            "rouge_l": round(rl, 4),
            "bert_score": round(bert, 4),
        })
        print(f"{sample.get('speaker_id'):<30} WER: {w:.3f} | BLEU: {b:.3f} | ROUGE-L: {rl:.3f} | BERT: {bert:.3f}")

    if not metrics_rows:
        print("No valid samples evaluated.")
        return

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_OUT.open("w") as f:
        json.dump(metrics_rows, f, indent=2)
    print(f"\nSaved metrics to {METRICS_OUT}")

    PER_SPEAKER_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PER_SPEAKER_CSV.open("w") as f:
        f.write("speaker_id,sample_id,WER,BLEU,ROUGE_L,BERTScore\n")
        for r in metrics_rows:
            f.write(f"{r['speaker_id']},{r['sample_id']},{r['wer']},{r['bleu']},{r['rouge_l']},{r['bert_score']}\n")
    print(f"Saved per-speaker CSV to {PER_SPEAKER_CSV}")


if __name__ == "__main__":
    main()
