"""
MTS-Dialog: condition-level metrics (WER, BLEU, ROUGE-L, NER).
When scispaCy was used in mts_eval, adds CHEMICAL F1 and DISEASE F1 (biomedical entity preservation).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results") / "mts" / "metrics"


def _safe_float(m, key):
    try:
        v = m.get(key)
        if v is None:
            return np.nan
        f = float(v)
        return np.nan if (f != f) else f
    except (TypeError, ValueError, KeyError):
        return np.nan


def main():
    summary_path = RESULTS_DIR / "mts_metrics_summary.json"
    if not summary_path.exists():
        print("No mts_metrics_summary.json. Run analysis/mts_eval first.")
        return

    with summary_path.open("r") as f:
        metrics = json.load(f)

    cond_order = ["c1", "c2a", "c2b", "c3_lex_gen", "c3_lex_rel", "c3_lex_mat", "c4_den_gen", "c4_den_rel", "c4_den_mat"]
    labels = {
        "c1": "C1 Noisy",
        "c2a": "C2a LLaMA",
        "c2b": "C2b Mistral",
        "c3_lex_gen": "C3-Lex-Gen",
        "c3_lex_rel": "C3-Lex-Rel",
        "c3_lex_mat": "C3-Lex-Mat",
        "c4_den_gen": "C4-Den-Gen",
        "c4_den_rel": "C4-Den-Rel",
        "c4_den_mat": "C4-Den-Mat",
    }
    cond_order = [c for c in cond_order if c in metrics and isinstance(metrics.get(c), dict)]
    if not cond_order:
        print("No valid metrics. Run analysis/mts_eval first.")
        return

    xs = np.arange(len(cond_order))
    xticklabels = [labels[c] for c in cond_order]
    wer = [metrics[c]["wer"] for c in cond_order]
    bleu = [metrics[c]["bleu"] for c in cond_order]
    rouge_l = [metrics[c]["rouge_l"] for c in cond_order]
    bert = [metrics[c]["bert"] for c in cond_order]
    def _ner_val(c):
        m = metrics.get(c, {})
        v = _safe_float(m, "ner_f1")
        if np.isnan(v):
            v = _safe_float(m, "scispacy_overall_f1")  # fallback
        return v
    ner = [_ner_val(c) for c in cond_order]
    ner_available = any(not np.isnan(x) for x in ner)

    # scispaCy: CHEMICAL F1, DISEASE F1 (from en_ner_bc5cdr_md)
    chemical = [_safe_float(metrics.get(c, {}), "chemical_f1") for c in cond_order]
    disease = [_safe_float(metrics.get(c, {}), "disease_f1") for c in cond_order]
    scispacy_available = any(not np.isnan(x) for x in chemical + disease)

    if scispacy_available:
        nrows, ncols = 3, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 11))
        fig.suptitle("MTS-Dialog: Condition-level Metrics (incl. scispaCy biomedical NER)", fontsize=13, fontweight="bold")
    else:
        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8))
        fig.suptitle("MTS-Dialog: Condition-level Metrics", fontsize=14, fontweight="bold")

    def _bar(ax, values, color, ylabel, title, ymax=1.0):
        plot_vals = [0.0 if np.isnan(v) else v for v in values]
        bar_width = max(0.5, 0.85 / max(1, len(cond_order)))
        ax.bar(xs, plot_vals, width=bar_width, color=color)
        ax.set_xticks(xs)
        ax.set_xticklabels(xticklabels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, ymax if not plot_vals else min(ymax, max(plot_vals) * 1.15))
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Row 0: WER — cap at 1.5 so C4-Den-Mat (verbosity-inflated) doesn't dwarf others
    wer_cap = 1.5
    bar_w = max(0.5, 0.85 / max(1, len(cond_order)))
    wer_plot = [min(w, wer_cap) for w in wer]
    axes[0, 0].bar(xs, wer_plot, width=bar_w, color="#3A6EA5")
    axes[0, 0].set_xticks(xs)
    axes[0, 0].set_xticklabels(xticklabels, rotation=30, ha="right")
    axes[0, 0].set_ylabel("WER (↓)")
    axes[0, 0].set_ylim(0.0, wer_cap * 1.1)
    c4_idx = next((i for i, c in enumerate(cond_order) if c == "c4_den_mat"), None)
    if c4_idx is not None and wer[c4_idx] > wer_cap:
        axes[0, 0].set_title("Word Error Rate\n(C4-Den-Mat 9.8 clipped; verbosity inflates WER)")
    else:
        axes[0, 0].set_title("Word Error Rate")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.4)
    _bar(axes[0, 1], bleu, "#27AE60", "BLEU (↑)", "BLEU")
    # Row 1: ROUGE-L, NER F1 (spaCy)
    _bar(axes[1, 0], rouge_l, "#E67E22", "ROUGE-L (↑)", "ROUGE-L")
    ax = axes[1, 1]
    if ner_available:
        _bar(ax, ner, "#8E44AD", "NER F1 (↑)", "Entity preservation (spaCy F1)")
    else:
        ax.set_xticks(xs)
        ax.set_xticklabels(xticklabels, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("NER F1 (↑)")
        ax.set_title("Entity preservation (spaCy F1)")
        ax.text(0.5, 0.5, "NER skipped\n(spaCy unavailable)",
                ha="center", va="center", fontsize=10, transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    if scispacy_available:
        # Row 2: scispaCy CHEMICAL F1, DISEASE F1
        _bar(axes[2, 0], chemical, "#16A085", "F1 (↑)", "CHEMICAL (scispaCy)")
        _bar(axes[2, 1], disease, "#C0392B", "F1 (↑)", "DISEASE (scispaCy)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path("images") / "mts_cond_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

