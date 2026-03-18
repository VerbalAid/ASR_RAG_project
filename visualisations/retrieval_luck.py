"""Per-speaker WER variability across TED retrieval conditions (scatter + trajectories)."""
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

# Professional styling
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

TED_METRICS_DIR = Path("results") / "ted" / "metrics"


def load_by_speaker(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["speaker_id"]: float(r["wer"]) for r in rows}


raw_asr_map = load_by_speaker(TED_METRICS_DIR / "c1_metrics.json")
llm_only_map = load_by_speaker(TED_METRICS_DIR / "c2_metrics_mistral.json")
c3_lex_gen_map = load_by_speaker(TED_METRICS_DIR / "c3_lex_gen_metrics.json")
c4_den_gen_map = load_by_speaker(TED_METRICS_DIR / "c4_den_gen_metrics.json")
c4_den_mat_map = load_by_speaker(TED_METRICS_DIR / "c4_den_mat_metrics.json")

speakers = sorted(
    set(raw_asr_map)
    & set(llm_only_map)
    & set(c3_lex_gen_map)
    & set(c4_den_gen_map)
    & set(c4_den_mat_map)
)

raw_asr = [raw_asr_map[s] for s in speakers]
llm_only = [llm_only_map[s] for s in speakers]
c3_lex_gen = [c3_lex_gen_map[s] for s in speakers]
c4_den_gen = [c4_den_gen_map[s] for s in speakers]
c4_den_mat = [c4_den_mat_map[s] for s in speakers]

# Abbreviate speaker names for x-axis
def abbreviate(name: str) -> str:
    parts = name.replace("_", " ").split()
    if len(parts) <= 1:
        return name
    return f"{parts[0][0]}.{parts[-1]}"


abbrev = [abbreviate(s) for s in speakers]

n_speakers = len(speakers)
# Slight jitter: separate x-positions for each condition so points don't overlap
x_base = np.arange(n_speakers)
jitter = 0.15
x_lex = x_base - jitter
x_dense = x_base
x_dom = x_base + jitter

fig, ax = plt.subplots(figsize=(11, 6))

# Faint grey lines connecting same speaker across the three conditions
for i in range(n_speakers):
    ax.plot(
        [x_lex[i], x_dense[i], x_dom[i]],
        [c3_lex_gen[i], c4_den_gen[i], c4_den_mat[i]],
        color="grey",
        alpha=0.35,
        linewidth=1,
        zorder=0,
    )

# Scatter: C3-Lex-Gen, C4-Den-Gen, C4-Den-Mat
ax.scatter(
    x_lex,
    c3_lex_gen,
    c="#e67e22",
    marker="o",
    s=70,
    label="C3-Lex-Gen",
    edgecolors="black",
    linewidths=0.5,
    zorder=2,
)
ax.scatter(
    x_dense,
    c4_den_gen,
    c="#3498db",
    marker="s",
    s=70,
    label="C4-Den-Gen",
    edgecolors="black",
    linewidths=0.5,
    zorder=2,
)
ax.scatter(
    x_dom,
    c4_den_mat,
    c="#2ecc71",
    marker="^",
    s=70,
    label="C4-Den-Mat",
    edgecolors="black",
    linewidths=0.5,
    zorder=2,
)

ax.set_xticks(x_base)
ax.set_xticklabels(abbrev, rotation=45, ha="right")
ax.set_ylabel("WER", fontsize=11)
ax.set_xlabel("Speaker", fontsize=11)
ax.set_title("Per-Speaker WER Variability Across Retrieval Conditions", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper right", fontsize=10)
ax.set_xlim(-0.7, n_speakers - 0.3)

plt.tight_layout()
plt.savefig("images/retrieval_luck.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: images/retrieval_luck.png")
