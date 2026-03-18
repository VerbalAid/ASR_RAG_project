"""WER improvement vs TED LLM-only baseline (from TED metric files)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
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

# Canonical TED metrics files
TED_METRICS_DIR = Path("results") / "ted" / "metrics"

COND_ORDER = [
    "c2b", "c1", "c2a", "c3_lex_gen", "c3_lex_rel", "c3_lex_mat",
    "c4_den_gen", "c4_den_mat",
]
LABELS = {
    "c1": "C1 Raw",
    "c2a": "C2a LLaMA",
    "c2b": "C2b LLM-Only",
    "c3_lex_gen": "C3-Lex-Gen",
    "c3_lex_rel": "C3-Lex-Rel",
    "c3_lex_mat": "C3-Lex-Mat",
    "c4_den_gen": "C4-Den-Gen",
    "c4_den_mat": "C4-Den-Mat",
}
FILE_MAP = {
    "c1": "c1_metrics.json",
    "c2a": "c2_metrics_llama.json",
    "c2b": "c2_metrics_mistral.json",
    "c3_lex_gen": "c3_lex_gen_metrics.json",
    "c3_lex_rel": "c3_lex_rel_metrics.json",
    "c3_lex_mat": "c3_lex_mat_metrics.json",
    "c4_den_gen": "c4_den_gen_metrics.json",
    "c4_den_mat": "c4_den_mat_metrics.json",
}


def load_metrics(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_wer_over_speakers(rows: list, speaker_ids: set = None) -> float:
    if speaker_ids is not None:
        rows = [r for r in rows if r.get("speaker_id") in speaker_ids]
    if not rows:
        return np.nan
    return float(np.mean([r["wer"] for r in rows]))


# Load baseline (C2b) once for speaker-matched comparisons
c2b_path = TED_METRICS_DIR / FILE_MAP["c2b"]
c2b_rows = load_metrics(c2b_path) if c2b_path.exists() else []

# Build conditions and WER from available files.
# C3-Lex-Mat has n=9 (EricMead excluded); use speaker-matched baseline for fair comparison.
conditions = []
wer = []
for c in COND_ORDER:
    p = TED_METRICS_DIR / FILE_MAP[c]
    if not p.exists():
        continue
    rows = load_metrics(p)
    cond_mean = mean_wer_over_speakers(rows)
    conditions.append(LABELS[c])
    wer.append(cond_mean)

# For improvement: use baseline over same speakers as each condition.
# C3-Lex-Mat has n=9 (EricMead excluded); use speaker-matched baseline for fair comparison.
cond_ids_with_data = [c for c in COND_ORDER if (TED_METRICS_DIR / FILE_MAP[c]).exists()]
improvement_pct = []
for i, (cond_id, cond_wer) in enumerate(zip(cond_ids_with_data, wer)):
    if i == 0:  # baseline
        improvement_pct.append(0.0)
        continue
    cond_path = TED_METRICS_DIR / FILE_MAP[cond_id]
    cond_rows = load_metrics(cond_path)
    cond_speakers = {r["speaker_id"] for r in cond_rows}
    baseline_matched = mean_wer_over_speakers(c2b_rows, cond_speakers)
    if baseline_matched > 0 and not np.isnan(cond_wer):
        imp = ((baseline_matched - cond_wer) / baseline_matched) * 100
        improvement_pct.append(imp)
    else:
        improvement_pct.append(0.0)

# Colours: grey baseline, green improvement, red degradation
def _color(val, is_baseline):
    if is_baseline:
        return "#7f8c8d"
    return "#2ecc71" if val >= 0 else "#e74c3c"

colors = [_color(improvement_pct[i], i == 0) for i in range(len(conditions))]

fig, ax = plt.subplots(figsize=(max(9, len(conditions) * 1.1), 5))
ax.set_title("WER Improvement Relative to LLM-Only Baseline (TED-LIUM)", fontsize=13, fontweight="bold")

x_pos = np.arange(len(conditions))
bar_width = max(0.4, 0.8 / max(1, len(conditions)))
bars = ax.bar(x_pos, improvement_pct, width=bar_width, color=colors, edgecolor="black", linewidth=0.5)

ax.axhline(y=0, color="black", linewidth=0.8)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Value labels on bars
for i, (bar, val) in enumerate(zip(bars, improvement_pct)):
    label = f"{val:+.0f}%" if not np.isnan(val) else "—"
    if val >= 0:
        y_pos = bar.get_height() + 5
        va = "bottom"
        t = ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                    ha="center", va=va, fontsize=11, fontweight="bold")
    else:
        # Inside bar to avoid x-axis clash; white text with outline for contrast
        y_pos = val * 0.4
        t = ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                    ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        t.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground="black")])
    t.set_clip_on(False)

ax.set_xticks(x_pos)
ax.set_xticklabels(conditions, rotation=35, ha="right")
ax.set_ylabel("Improvement (%)", fontsize=11)
ax.set_xlabel("Condition", fontsize=11)
ax.set_ylim(-125, 90)
ax.set_xlim(-0.6, len(conditions) - 0.4)

plt.tight_layout()
plt.savefig("images/improvement_chart.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/improvement_chart.png")
