"""Evaluate C1, C2b, C4-Den-Mat on full 57; write metrics and Wilcoxon."""
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from bert_score import score as bert_score
from jiwer import wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "primock57_full_57"
METRICS_DIR = RESULTS_DIR / "metrics"
PER_DIALOGUE_DIR = RESULTS_DIR / "per_dialogue"

CONDITION_CONFIG = [
    ("c1", "c1_whisper_raw.json", "c1_whisper_tiny"),
    ("c2b", "c2b_outputs.json", "c2b_mistral"),
    ("c4_den_mat", "c4_den_mat_outputs.json", "c4_den_mat"),
    ("c4_den_gen", "c4_den_gen_outputs.json", "c4_den_gen"),  # optional
]


def extract_ollama_content(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    if "model=" not in raw and "message=" not in raw:
        return raw.strip()
    m = re.search(r'content="((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    if m:
        return m.group(1).replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\").strip()
    m2 = re.search(r"content='((?:[^'\\]|\\.)*)'", raw, re.DOTALL)
    if m2:
        return m2.group(1).replace("\\n", "\n").replace("\\'", "'").replace("\\\\", "\\").strip()
    return raw.strip()


def normalise(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_metrics(ref: str, hyp: str, scorer, smoothing) -> dict:
    ref_n = normalise(ref)
    hyp_n = normalise(hyp)
    if not hyp_n:
        return {"wer": 1.0, "bleu": 0.0, "rouge_l": 0.0, "bert_score": 0.0}
    w = wer(ref_n, hyp_n)
    b = sentence_bleu([ref_n.split()], hyp_n.split(), smoothing_function=smoothing)
    r = scorer.score(ref_n, hyp_n)["rougeL"].fmeasure
    _, _, f1 = bert_score([hyp_n], [ref_n], lang="en", verbose=False)
    bert = float(f1.item())
    return {"wer": w, "bleu": b, "rouge_l": r, "bert_score": bert}


def run_wilcoxon_two_sided(a: List[float], b: List[float]):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) < 2:
        return None, None
    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0
    try:
        res = wilcoxon(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return None, None


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PER_DIALOGUE_DIR.mkdir(parents=True, exist_ok=True)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    c1_path = RESULTS_DIR / "c1_whisper_raw.json"
    if not c1_path.exists():
        print(f"Error: Missing {c1_path}. Run c1_whisper_57.py first.", file=sys.stderr)
        sys.exit(1)

    with c1_path.open(encoding="utf-8") as f:
        c1_data = json.load(f)

    summary: Dict[str, Dict[str, float]] = {}
    per_sample_metrics: Dict[str, List[Dict[str, Any]]] = {}

    for cond, filename, hyp_key in CONDITION_CONFIG:
        path = RESULTS_DIR / filename
        if not path.exists():
            if cond == "c4_den_gen":
                continue
            print(f"Error: Missing {path}. Run run_c2b_c4.py first.", file=sys.stderr)
            sys.exit(1)

        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for sample in data:
            ref = sample.get("ground_truth", "")
            hyp_raw = sample.get(hyp_key, "")
            hyp = extract_ollama_content(hyp_raw) or hyp_raw
            if not ref:
                continue
            m = compute_metrics(ref, hyp, scorer, smoothing)
            row = {
                "sample_id": sample.get("sample_id"),
                "speaker_id": sample.get("speaker_id"),
                "wer": round(m["wer"], 4),
                "bleu": round(m["bleu"], 4),
                "rouge_l": round(m["rouge_l"], 4),
                "bert_score": round(m["bert_score"], 4),
            }
            rows.append(row)

        per_sample_metrics[cond] = rows
        summary[cond] = {
            "wer": float(np.mean([r["wer"] for r in rows])),
            "bleu": float(np.mean([r["bleu"] for r in rows])),
            "rouge_l": float(np.mean([r["rouge_l"] for r in rows])),
            "bert_score": float(np.mean([r["bert_score"] for r in rows])),
            "n": len(rows),
        }

        csv_path = PER_DIALOGUE_DIR / f"primock57_full_57_{cond}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "speaker_id", "wer", "bleu", "rouge_l", "bert_score"])
            w.writeheader()
            w.writerows(rows)
        print(f"  Wrote {csv_path} ({len(rows)} rows)")

    summary_path = METRICS_DIR / "primock57_full_57_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {summary_path}")

    # Wilcoxon: C4-Den-Mat vs C2b (and optionally C4-Den-Gen vs C2b)
    if "c2b" not in per_sample_metrics or "c4_den_mat" not in per_sample_metrics:
        print("  Skip Wilcoxon: missing c2b or c4_den_mat.")
    else:
        c2b_rows = {r["sample_id"]: r for r in per_sample_metrics["c2b"]}
        c4_rows = {r["sample_id"]: r for r in per_sample_metrics["c4_den_mat"]}
        common_ids = sorted(set(c2b_rows) & set(c4_rows))
        lines = [
            "# Wilcoxon signed-rank (PriMock57 full-57)",
            "",
            f"n = {len(common_ids)} consultations. Two-tailed, α = 0.05.",
            "",
            "## C4-Den-Mat vs C2b (core claim)",
            "",
            "| Metric | C2b mean | C4-Den-Mat mean | Statistic | p-value | Significant (α=0.05) |",
            "|--------|----------|-----------------|-----------|---------|----------------------|",
        ]

        for metric in ["wer", "bleu", "rouge_l", "bert_score"]:
            a = [c2b_rows[i][metric] for i in common_ids]
            b = [c4_rows[i][metric] for i in common_ids]
            stat, p = run_wilcoxon_two_sided(a, b)
            sig = "Yes" if p is not None and p < 0.05 else "No"
            stat_s = f"{stat:.4f}" if stat is not None else "—"
            p_s = f"{p:.4f}" if p is not None else "—"
            a_mean = np.mean(a)
            b_mean = np.mean(b)
            lines.append(f"| {metric} | {a_mean:.4f} | {b_mean:.4f} | {stat_s} | {p_s} | {sig} |")

        if "c4_den_gen" in per_sample_metrics:
            gen_rows = {r["sample_id"]: r for r in per_sample_metrics["c4_den_gen"]}
            common_gen = sorted(set(c2b_rows) & set(gen_rows))
            lines.extend([
                "",
                "## C4-Den-Gen vs C2b",
                "",
                "| Metric | C2b mean | C4-Den-Gen mean | Statistic | p-value | Significant (α=0.05) |",
                "|--------|----------|-----------------|-----------|---------|----------------------|",
            ])
            for metric in ["wer", "bleu", "rouge_l", "bert_score"]:
                a = [c2b_rows[i][metric] for i in common_gen]
                b = [gen_rows[i][metric] for i in common_gen]
                stat, p = run_wilcoxon_two_sided(a, b)
                sig = "Yes" if p is not None and p < 0.05 else "No"
                stat_s = f"{stat:.4f}" if stat is not None else "—"
                p_s = f"{p:.4f}" if p is not None else "—"
                a_mean = np.mean(a)
                b_mean = np.mean(b)
                lines.append(f"| {metric} | {a_mean:.4f} | {b_mean:.4f} | {stat_s} | {p_s} | {sig} |")

        wilcoxon_path = METRICS_DIR / "primock57_full_57_wilcoxon.md"
        wilcoxon_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Wrote {wilcoxon_path}")
        print("")
        print("--- Wilcoxon C4-Den-Mat vs C2b ---")
        for line in lines[6:6+6]:
            print(line)


if __name__ == "__main__":
    main()
