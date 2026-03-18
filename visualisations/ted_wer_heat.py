import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results") / "ted" / "metrics"


def load_metrics(path: Path):
    with path.open("r") as f:
        return json.load(f)


def main():
    c1 = load_metrics(RESULTS_DIR / "c1_metrics.json")
    c2b = load_metrics(RESULTS_DIR / "c2_metrics_mistral.json")
    c3_lex_gen = load_metrics(RESULTS_DIR / "c3_lex_gen_metrics.json")
    c4_den_gen = load_metrics(RESULTS_DIR / "c4_den_gen_metrics.json")
    c4_den_mat = load_metrics(RESULTS_DIR / "c4_den_mat_metrics.json")

    speakers = [r["speaker_id"] for r in c1]

    def wer_vec(records):
        lu = {r["speaker_id"]: r for r in records}
        return [lu[s]["wer"] if s in lu else float("nan") for s in speakers]

    rows = [wer_vec(c1), wer_vec(c2b), wer_vec(c3_lex_gen), wer_vec(c4_den_gen), wer_vec(c4_den_mat)]
    labels = ["C1 Raw", "C2b LLM", "C3-Lex-Gen", "C4-Den-Gen", "C4-Den-Mat"]
    if (RESULTS_DIR / "c3_lex_rel_metrics.json").exists():
        c3_lex_rel = load_metrics(RESULTS_DIR / "c3_lex_rel_metrics.json")
        rows.insert(4, wer_vec(c3_lex_rel))
        labels.insert(4, "C3-Lex-Rel")
    if (RESULTS_DIR / "c3_lex_mat_metrics.json").exists():
        c3_lex_mat = load_metrics(RESULTS_DIR / "c3_lex_mat_metrics.json")
        rows.insert(5 if (RESULTS_DIR / "c3_lex_rel_metrics.json").exists() else 4, wer_vec(c3_lex_mat))
        labels.insert(5 if (RESULTS_DIR / "c3_lex_rel_metrics.json").exists() else 4, "C3-Lex-Mat")

    wer_matrix = np.array(rows)

    fig, ax = plt.subplots(figsize=(14, max(4, len(rows) * 0.8)))
    im = ax.imshow(wer_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0.0, vmax=0.8)

    ax.set_xticks(range(len(speakers)))
    ax.set_xticklabels(speakers, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("TED-LIUM: WER Heatmap by Speaker and Condition")

    for i in range(wer_matrix.shape[0]):
        for j in range(wer_matrix.shape[1]):
            val = wer_matrix[i, j]
            if np.isnan(val):
                txt, color = "—", "black"
            else:
                txt, color = f"{val:.3f}", "white" if val > 0.4 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("WER")

    fig.tight_layout()
    out_path = Path("images") / "ted_wer_heat.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

