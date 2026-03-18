"""NER F1 vs BERTScore divergence on MTS (from mts_metrics_summary.json)."""
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

# Professional styling (clearer typography)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11

# Current MTS metrics
MTS_SUMMARY = Path("results") / "mts" / "metrics" / "mts_metrics_summary.json"
metrics = json.loads(MTS_SUMMARY.read_text(encoding="utf-8"))

# All MTS conditions (exclude note keys)
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
cond_ids = [c for c in COND_ORDER if c in metrics and isinstance(metrics.get(c), dict)]
conditions = [LABELS[c] for c in cond_ids]
bert_score = [metrics[c]["bert"] for c in cond_ids]


def _safe_ner(c):
    m = metrics.get(c, {})
    v = m.get("ner_f1") if isinstance(m, dict) else None
    if v is None:
        v = m.get("scispacy_overall_f1") if isinstance(m, dict) else None  # fallback
    if v is None:
        return np.nan
    try:
        f = float(v)
        return np.nan if (f != f) else f
    except (TypeError, ValueError):
        return np.nan


ner_f1 = [_safe_ner(c) for c in cond_ids]
ner_available = any(not np.isnan(x) for x in ner_f1)
if not ner_available:
    ner_f1 = [0.0] * len(cond_ids)


def _ner_label():
    for c in cond_ids:
        m = metrics.get(c, {})
        if isinstance(m, dict) and (m.get("scispacy_overall_f1") is not None or m.get("chemical_f1") is not None):
            return "NER F1 (scispaCy)"
    return "NER F1"

# Bar colors: BERTScore and NER F1 (high contrast)
metric_colors = {"BERTScore": "#2980B9", "NER F1": "#D35400"}

x = np.arange(len(conditions))
width = 0.42
n_conds = len(conditions)
fig, ax = plt.subplots(figsize=(max(10, n_conds * 1.1), 5.5))
fig.suptitle("Semantic Preservation vs Entity Fidelity", fontsize=19, fontweight="bold", y=1.0)
fig.text(0.5, 0.90, "MTS-Dialog (n=100)  |  Higher is better", ha="center", fontsize=11, color="#444444")
bars1 = ax.bar(x - width / 2, bert_score, width, label="BERTScore", color=metric_colors["BERTScore"], edgecolor="#1a5276", linewidth=1)
bars2 = ax.bar(x + width / 2, ner_f1, width, label=_ner_label(), color=metric_colors["NER F1"], edgecolor="#a04000", linewidth=1)

# Value labels on bars
def add_labels(bars, x_offset=0):
    for bar in bars:
        h = bar.get_height()
        t = ax.text(
            bar.get_x() + bar.get_width() / 2 + x_offset,
            h + 0.02,
            f"{h:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
        t.set_clip_on(False)

add_labels(bars1)
add_labels(bars2, x_offset=0.05)

ax.set_xticks(x)
ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=11)
ax.set_ylabel("Score", fontsize=13, fontweight="bold")
ax.set_xlabel("Condition", fontsize=13, fontweight="bold")
# Y-axis: show full range when NER has real data; otherwise focus on BERTScore
valid_bert = [v for v in bert_score if not (isinstance(v, float) and np.isnan(v))]
valid_ner = [v for v in ner_f1 if v > 0]  # exclude zeros from "unavailable" placeholder
if valid_ner:
    y_max = max((max(valid_bert) if valid_bert else 0), max(valid_ner), 0.3)
    ax.set_ylim(0, min(1.08, y_max * 1.15))
else:
    ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="-", alpha=0.35)
ax.set_axisbelow(True)
if not ner_available:
    ax.text(0.5, 0.95, "NER F1 unavailable. Run: python -m analysis.mts_eval (requires spaCy or scispaCy)", ha="center", va="top",
            fontsize=9, transform=ax.transAxes, style="italic")
# Reserve right margin for legend (top-right, outside plot)
fig.tight_layout(rect=[0, 0.08, 0.75, 0.88])
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=13, framealpha=0.95)
ax.set_xlim(-0.7, n_conds - 0.3)
plt.savefig("images/ner_bert_divergence.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/ner_bert_divergence.png")
