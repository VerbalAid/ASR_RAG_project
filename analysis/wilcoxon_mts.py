"""Wilcoxon signed-rank tests on MTS per-dialogue metrics; writes .json, .md, .tex."""
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "mts"
PER_DIALOGUE_DIR = RESULTS_DIR / "per_dialogue"
METRICS_DIR = RESULTS_DIR / "metrics"

# Comparisons: (cond_a, cond_b, label) — MTS n=100
COMPARISONS = [
    ("c4_den_mat", "c2b", "C4-Den-Mat vs C2b"),
    ("c4_den_gen", "c3_lex_gen", "C4-Den-Gen vs C3-Lex-Gen (modality, same AG News corpus)"),
    ("c3_lex_mat", "c3_lex_gen", "C3-Lex-Mat vs C3-Lex-Gen (corpus effect, BM25)"),
    ("c4_den_mat", "c3_lex_mat", "C4-Den-Mat vs C3-Lex-Mat (modality, same domain-matched corpus)"),
    ("c4_den_rel", "c3_lex_rel", "C4-Den-Rel vs C3-Lex-Rel (modality, same PriMock57 corpus)"),
    ("c4_den_mat", "c4_den_gen", "C4-Den-Mat vs C4-Den-Gen (corpus effect, dense)"),
    ("c3_lex_rel", "c3_lex_gen", "C3-Lex-Rel vs C3-Lex-Gen (corpus effect, BM25)"),
]

METRICS = ["WER", "BLEU", "ROUGE_L", "BERTScore", "NER_F1"]
OPTIONAL_METRICS = ["CHEMICAL_F1", "DISEASE_F1", "SCISPACY_OVERALL_F1"]
ALPHA = 0.05


def load_per_dialogue_csv(cond: str) -> dict:
    """Load per-dialogue CSV for condition; return dict dialogue_id -> {metric: value}."""
    path = PER_DIALOGUE_DIR / f"mts_per_dialogue_{cond}.csv"
    if not path.exists():
        return {}
    out = {}
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = row.get("dialogue_id", "").strip()
            if not did:
                continue
            try:
                did = int(did)
            except ValueError:
                continue
            out[did] = {}
            for key in list(row):
                if key == "dialogue_id":
                    continue
                try:
                    val = float(row[key])
                except (ValueError, TypeError):
                    val = np.nan
                out[did][key] = val
    return out


def get_paired_arrays(data_a: dict, data_b: dict, metric: str):
    """Return (arr_a, arr_b) aligned by dialogue_id; exclude if either is missing or NaN."""
    common = sorted(set(data_a.keys()) & set(data_b.keys()))
    arr_a = []
    arr_b = []
    for did in common:
        va, vb = data_a[did].get(metric, np.nan), data_b[did].get(metric, np.nan)
        if np.isnan(va) or np.isnan(vb):
            continue
        arr_a.append(va)
        arr_b.append(vb)
    return np.array(arr_a), np.array(arr_b)


def run_wilcoxon(arr_a, arr_b):
    """Two-tailed Wilcoxon; return (statistic, p_value)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diff = arr_a - arr_b
        if len(diff) < 3 or np.all(diff == 0):
            return np.nan, 1.0
        try:
            res = wilcoxon(arr_a, arr_b, alternative="two-sided")
            return res.statistic, res.pvalue
        except Exception:
            return np.nan, np.nan


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = METRICS.copy()
    # Add optional metrics if present in any CSV
    for cond in ["c4_den_mat", "c2b", "c4_den_gen", "c4_den_rel", "c3_lex_gen", "c3_lex_rel", "c3_lex_mat", "c2a"]:
        d = load_per_dialogue_csv(cond)
        if d:
            first = next(iter(d.values()), {})
            for m in OPTIONAL_METRICS:
                if m in first and m not in all_metrics:
                    all_metrics.append(m)
            break

    rows = []
    for cond_a, cond_b, label in COMPARISONS:
        data_a = load_per_dialogue_csv(cond_a)
        data_b = load_per_dialogue_csv(cond_b)
        if not data_a or not data_b:
            for m in all_metrics:
                rows.append((label, m, np.nan, "N/A (missing data)"))
            continue
        for metric in all_metrics:
            arr_a, arr_b = get_paired_arrays(data_a, data_b, metric)
            if len(arr_a) < 3:
                rows.append((label, metric, np.nan, "N/A (insufficient data)"))
                continue
            stat, p = run_wilcoxon(arr_a, arr_b)
            if np.isnan(p):
                sig = "N/A"
            elif p < 0.01:
                sig = "Yes (p<0.01)"
            elif p < ALPHA:
                sig = "Yes (p<0.05)"
            else:
                sig = "No"
            rows.append((label, metric, p, sig))

    # Markdown table
    lines = [
        "",
        "## MTS-Dialog Wilcoxon signed-rank test (per-dialogue, α=0.05)",
        "",
        "| Comparison | Metric | p-value | Significant |",
        "|------------|--------|---------|--------------|",
    ]
    for label, metric, p, sig in rows:
        p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
        lines.append(f"| {label} | {metric} | {p_str} | {sig} |")

    md_path = METRICS_DIR / "mts_wilcoxon_results.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path.read_text())

    # JSON output
    json_rows = [{"comparison": label, "metric": m, "p_value": float(p) if not np.isnan(p) else None, "significant": sig} for label, m, p, sig in rows]
    json_path = METRICS_DIR / "mts_wilcoxon_results.json"
    with json_path.open("w") as f:
        json.dump({"comparisons": json_rows, "alpha": ALPHA, "n": 100}, f, indent=2)
    print(f"\nSaved JSON to {json_path}")

    # LaTeX table
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{MTS-Dialog: Wilcoxon signed-rank test (per-dialogue, two-tailed, $\alpha = 0.05$).}",
        r"\label{tab:mts-wilcoxon}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Comparison & Metric & $p$-value & Sig. \\",
        r"\midrule",
    ]
    for label, metric, p, sig in rows:
        p_str = f"{p:.4f}" if not np.isnan(p) else "---"
        sig_tex = sig.replace("Yes (p<0.01)", r"Yes ($p<0.01$)").replace("Yes (p<0.05)", r"Yes ($p<0.05$)")
        tex_lines.append(f"{label} & {metric} & {p_str} & {sig_tex} \\\\")
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_path = METRICS_DIR / "mts_wilcoxon_results.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"\nSaved LaTeX table to {tex_path}")


if __name__ == "__main__":
    main()
