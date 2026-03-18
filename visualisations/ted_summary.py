import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results") / "ted" / "metrics"


def load_metrics(path: Path):
    with path.open("r") as f:
        return json.load(f)


def compute_averages(records, prefix: str = ""):
    wer = np.mean([r["wer"] for r in records])
    bleu = np.mean([r["bleu"] for r in records])
    rouge_l = np.mean([r["rouge_l"] for r in records])
    bert = np.mean([r["bert_score"] for r in records])
    return {
        "label": prefix,
        "wer": wer,
        "bleu": bleu,
        "rouge_l": rouge_l,
        "bert": bert,
    }


def main():
    # Load all available TED metrics
    metrics_files = [
        ("c1_metrics.json", "C1 Raw"),
        ("c2_metrics_mistral.json", "C2b LLM"),
        ("c2_metrics_llama.json", "C2a LLaMA"),
        ("c3_lex_gen_metrics.json", "C3-Lex-Gen"),
        ("c3_lex_rel_metrics.json", "C3-Lex-Rel"),
        ("c3_lex_mat_metrics.json", "C3-Lex-Mat"),
        ("c4_den_gen_metrics.json", "C4-Den-Gen"),
        ("c4_den_mat_metrics.json", "C4-Den-Mat"),
    ]
    conds = []
    for fname, label in metrics_files:
        p = RESULTS_DIR / fname
        if p.exists():
            data = load_metrics(p)
            conds.append(compute_averages(data, label))

    labels = [c["label"] for c in conds]
    wer = [c["wer"] for c in conds]
    bleu = [c["bleu"] for c in conds]
    rouge_l = [c["rouge_l"] for c in conds]
    bert = [c["bert"] for c in conds]

    x = np.arange(len(labels))
    width = max(0.35, 0.85 / len(labels))
    fig_w = max(10, len(labels) * 1.2)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 8))
    fig.suptitle("TED-LIUM: Condition-level Metrics", fontsize=14, fontweight="bold")

    # WER (lower is better)
    ax = axes[0, 0]
    ax.bar(x, wer, width=width, color="#3A6EA5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("WER (↓)")
    ax.set_ylim(0.0, max(wer) * 1.2)
    ax.set_title("Word Error Rate")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # BLEU
    ax = axes[0, 1]
    ax.bar(x, bleu, width=width, color="#27AE60")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("BLEU (↑)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("BLEU")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ROUGE-L
    ax = axes[1, 0]
    ax.bar(x, rouge_l, width=width, color="#E67E22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("ROUGE-L (↑)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("ROUGE-L")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # BERTScore
    ax = axes[1, 1]
    ax.bar(x, bert, width=width, color="#8E44AD")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("BERTScore (↑)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("BERTScore")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = Path("images") / "ted_cond_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

