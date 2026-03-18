"""Bar chart of PriMock57 speech metrics per condition; writes images/primock57_speech_cond_metrics.png."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RESULTS_DIR = Path("results") / "primock57_speech" / "metrics"
COND_ORDER = [
    "c1", "c2a", "c2b",
    "c3_lex_gen", "c3_lex_rel", "c3_lex_mat",
    "c4_den_gen", "c4_den_mat",
]
LABELS = {
    "c1": "C1 Whisper",
    "c2a": "C2a Llama",
    "c2b": "C2b Mistral",
    "c3_lex_gen": "C3-Lex-Gen",
    "c3_lex_rel": "C3-Lex-Rel",
    "c3_lex_mat": "C3-Lex-Mat",
    "c4_den_gen": "C4-Den-Gen",
    "c4_den_mat": "C4-Den-Mat",
}
# Group colours: baseline (grey) → LLM-only (distinct) → Lexical RAG → Dense RAG
GROUP_COLORS = {
    "c1": "#5D6D7E",    # Dark grey (baseline)
    "c2a": "#7F8C8D",   # Medium grey (LLaMA)
    "c2b": "#3498DB",   # Blue (Mistral - distinct from C1)
    "c3_lex_gen": "#E67E22",
    "c3_lex_rel": "#D35400",
    "c3_lex_mat": "#BA4A00",
    "c4_den_gen": "#2980B9",
    "c4_den_mat": "#1A5276",
}


def main():
    available = []
    for c in COND_ORDER:
        p = RESULTS_DIR / f"{c}_metrics.json"
        if p.exists():
            available.append(c)
    if not available:
        print("No metrics found. Run analysis/primock57_speech_eval (or primock57_speech_eval_with_ner) first.")
        return

    summary_path = RESULTS_DIR / "primock57_metrics_summary.json"
    summary = {}
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)

    def _safe_float(d, key):
        v = d.get(key)
        return float(v) if v is not None else np.nan

    def mean_from_file(cond, key):
        if cond in summary and key in summary[cond]:
            return _safe_float(summary[cond], key)
        p = RESULTS_DIR / f"{cond}_metrics.json"
        if not p.exists():
            return np.nan
        with p.open() as f:
            data = json.load(f)
        vals = [r.get(key) for r in data if key in r and r[key] is not None]
        return float(np.mean(vals)) if vals else np.nan

    xs = np.arange(len(available))
    wer = [mean_from_file(c, "wer") for c in available]
    bleu = [mean_from_file(c, "bleu") for c in available]
    rouge_l = [mean_from_file(c, "rouge_l") for c in available]
    bert = [mean_from_file(c, "bert_score") for c in available]
    chemical = [mean_from_file(c, "chemical_f1") for c in available]
    disease = [mean_from_file(c, "disease_f1") for c in available]
    xticklabels = [LABELS[c] for c in available]
    colors = [GROUP_COLORS.get(c, "#95A5A6") for c in available]

    scispacy_available = any(not np.isnan(x) for x in chemical + disease)

    if scispacy_available:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle("PriMock57 Speech: ASR → LLM → RAG (incl. scispaCy biomedical NER)", fontsize=16, fontweight="bold", y=0.98)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("PriMock57 Speech: ASR → LLM → RAG", fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.90,
        "Ground truth: Praat TextGrids  |  RAG: Wikipedia (Gen/Rel) + MTS-Dialog (Mat)  |  ↑ higher better except WER",
        ha="center", fontsize=10, color="#555555"
    )

    def _bar(ax, values, ylabel, title, lower_better=False, ymax=None):
        plot_vals = [0.0 if np.isnan(v) else v for v in values]
        bar_width = max(0.5, 0.85 / max(1, len(available)))
        ax.bar(xs, plot_vals, width=bar_width, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(xticklabels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        if lower_better and plot_vals:
            ax.set_ylim(0.0, max(plot_vals) * 1.2)
        else:
            # Use realistic fixed scales (0–0.5 for BLEU/ROUGE-L, 0–1 for F1) so full picture is visible
            if ymax is not None:
                ax.set_ylim(0.0, ymax)
            else:
                data_max = max(p for p in plot_vals if not np.isnan(p)) if plot_vals else 1.0
                ax.set_ylim(0.0, min(1.0, data_max * 1.15))
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    _bar(axes[0, 0], wer, "WER (↓)", "Word Error Rate (lower is better)", lower_better=True)
    _bar(axes[0, 1], bleu, "BLEU (↑)", "BLEU", ymax=0.5)
    _bar(axes[1, 0], rouge_l, "ROUGE-L (↑)", "ROUGE-L", ymax=0.5)
    _bar(axes[1, 1], bert, "BERTScore (↑)", "BERTScore", ymax=0.9)

    if scispacy_available:
        _bar(axes[2, 0], chemical, "F1 (↑)", "CHEMICAL (scispaCy)", ymax=0.5)
        _bar(axes[2, 1], disease, "F1 (↑)", "DISEASE (scispaCy)", ymax=0.9)

    legend_patches = [
        mpatches.Patch(color=GROUP_COLORS[c], label=LABELS[c])
        for c in available
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=12, framealpha=0.95)

    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
    out_path = Path("images") / "primock57_speech_cond_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
