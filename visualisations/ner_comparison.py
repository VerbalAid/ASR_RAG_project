"""Compare general vs biomedical NER on MTS; plot to images/ner_general_vs_scispacy.png."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

plt.rcParams["font.size"] = 11

MTS_SUMMARY = Path("results") / "mts" / "metrics" / "mts_metrics_summary.json"
COND_ORDER = ["c1", "c2a", "c2b", "c3_lex_gen", "c3_lex_rel", "c3_lex_mat", "c4_den_gen", "c4_den_rel", "c4_den_mat"]
LABELS = {
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


def _safe(m, key):
    v = m.get(key) if isinstance(m, dict) else None
    if v is None:
        return np.nan
    try:
        f = float(v)
        return np.nan if (f != f) else f
    except (TypeError, ValueError):
        return np.nan


def main():
    if not MTS_SUMMARY.exists():
        print("Run analysis/mts_eval first.")
        return

    metrics = json.loads(MTS_SUMMARY.read_text(encoding="utf-8"))
    cond_ids = [c for c in COND_ORDER if c in metrics and isinstance(metrics.get(c), dict)]
    if not cond_ids:
        print("No valid metrics.")
        return

    conditions = [LABELS[c] for c in cond_ids]
    ner_general = [_safe(metrics.get(c, {}), "ner_f1") for c in cond_ids]
    ner_scispacy = [_safe(metrics.get(c, {}), "scispacy_overall_f1") for c in cond_ids]

    general_ok = any(not np.isnan(x) for x in ner_general)
    scispacy_ok = any(not np.isnan(x) for x in ner_scispacy)

    if not general_ok and not scispacy_ok:
        print("No NER metrics. Run mts_eval with spaCy (en_core_web_sm) and/or scispaCy (en_ner_bc5cdr_md).")
        return

    if not general_ok:
        print("General NER missing. Install: pip install spacy && python -m spacy download en_core_web_sm")
    if not scispacy_ok:
        print("scispaCy NER missing. Install scispacy and en_ner_bc5cdr_md.")

    x = np.arange(len(conditions))
    width = 0.32
    n_conds = len(conditions)

    fig, ax = plt.subplots(figsize=(max(10, n_conds * 1.1), 5.5))
    fig.suptitle("NER F1: General (spaCy) vs Biomedical (scispaCy)", fontsize=16, fontweight="bold", y=1.0)
    fig.text(0.5, 0.92, "MTS-Dialog (n=100)  |  Compare entity fidelity under different NER models", ha="center", fontsize=11, color="#444444")

    bars = []
    if general_ok:
        vals = [0.0 if np.isnan(v) else v for v in ner_general]
        b1 = ax.bar(x - width / 2, vals, width, label="NER F1 (spaCy general)", color="#E67E22", edgecolor="#A04000")
        bars.append(b1)
    if scispacy_ok:
        vals = [0.0 if np.isnan(v) else v for v in ner_scispacy]
        b2 = ax.bar(x + width / 2, vals, width, label="NER F1 (scispaCy biomedical)", color="#16A085", edgecolor="#0E6655")
        bars.append(b2)

    def add_labels(bar_list):
        for barset in bar_list:
            for bar in barset:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    add_labels(bars)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=35, ha="right")
    ax.set_ylabel("NER F1 (↑)")
    ax.set_xlabel("Condition")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="-", alpha=0.35)

    fig.tight_layout(rect=[0, 0.08, 0.85, 0.88])
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=11, framealpha=0.95)

    out_path = Path("images") / "ner_general_vs_scispacy.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
