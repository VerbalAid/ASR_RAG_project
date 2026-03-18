"""PriMock57 speech eval with optional scispaCy NER (CHEMICAL/DISEASE F1)."""
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from bert_score import score as bert_score
from jiwer import wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "primock57_speech"
METRICS_DIR = RESULTS_DIR / "metrics"
PER_DIALOGUE_DIR = RESULTS_DIR / "per_dialogue"

CONDITION_FILES = {
    "c1": ("c1_whisper_raw.json", "c1_whisper_tiny"),
    "c2a": ("c2a_outputs.json", "c2a_llama"),
    "c2b": ("c2b_outputs.json", "c2b_mistral"),
    "c3_lex_gen": ("c3_lex_gen_outputs.json", "c3_lex_gen"),
    "c3_lex_rel": ("c3_lex_rel_outputs.json", "c3_lex_rel"),
    "c3_lex_mat": ("c3_lex_mat_outputs.json", "c3_lex_mat"),
    "c4_den_gen": ("c4_den_gen_outputs.json", "c4_den_gen"),
    "c4_den_mat": ("c4_den_mat_outputs.json", "c4_den_mat"),
}


def extract_ollama_content(raw: str) -> str:
    """Extract assistant content from Ollama response string."""
    if not raw or not isinstance(raw, str):
        return ""
    if "model=" not in raw and "message=" not in raw:
        return raw.strip()
    m = re.search(r'content="((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    if m:
        s = m.group(1).replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        return s.strip()
    m2 = re.search(r"content='((?:[^'\\]|\\.)*)'", raw, re.DOTALL)
    if m2:
        s = m2.group(1).replace("\\n", "\n").replace("\\'", "'").replace("\\\\", "\\")
        return s.strip()
    return raw.strip()


def normalise(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_ner_f1_by_type(
    nlp: Any, ref: str, hyp: str
) -> Tuple[float, float, float]:
    """CHEMICAL F1, DISEASE F1, overall F1 (scispaCy en_ner_bc5cdr_md)."""
    doc_ref = nlp(ref)
    doc_hyp = nlp(hyp)

    def _f1(ref_ents: set, hyp_ents: set) -> float:
        if not ref_ents and not hyp_ents:
            return 1.0
        if not ref_ents or not hyp_ents:
            return 0.0
        tp = len(ref_ents & hyp_ents)
        fp = len(hyp_ents - ref_ents)
        fn = len(ref_ents - hyp_ents)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    ref_by_label: Dict[str, set] = {}
    hyp_by_label: Dict[str, set] = {}
    for ent in doc_ref.ents:
        ref_by_label.setdefault(ent.label_, set()).add(ent.text.lower())
    for ent in doc_hyp.ents:
        hyp_by_label.setdefault(ent.label_, set()).add(ent.text.lower())

    chemical_f1 = _f1(
        ref_by_label.get("CHEMICAL", set()),
        hyp_by_label.get("CHEMICAL", set()),
    )
    disease_f1 = _f1(
        ref_by_label.get("DISEASE", set()),
        hyp_by_label.get("DISEASE", set()),
    )
    all_ref = set().union(*ref_by_label.values()) if ref_by_label else set()
    all_hyp = set().union(*hyp_by_label.values()) if hyp_by_label else set()
    overall_f1 = _f1(all_ref, all_hyp)
    return chemical_f1, disease_f1, overall_f1


def compute_metrics(
    ref: str, hyp: str, scorer, smoothing
) -> dict:
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


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PER_DIALOGUE_DIR.mkdir(parents=True, exist_ok=True)

    nlp_scispacy: Optional[Any] = None
    try:
        import spacy
        nlp_scispacy = spacy.load("en_ner_bc5cdr_md")
        print("Loaded scispaCy (en_ner_bc5cdr_md) for CHEMICAL/DISEASE F1.")
    except Exception as e:
        print(
            f"[primock57_eval] scispaCy not available: {e}\n"
            "  Install: pip install scispacy && pip install "
            "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
        )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    c1_path = RESULTS_DIR / "c1_whisper_raw.json"
    if not c1_path.exists():
        print(f"Error: Missing {c1_path}. Run PriMock57 speech C1 first.")
        return

    summary: Dict[str, Dict[str, float]] = {}

    for cond, (filename, hyp_key) in CONDITION_FILES.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"Skipping {cond}: {path} not found.")
            continue

        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        chemical_vals: List[float] = []
        disease_vals: List[float] = []
        overall_vals: List[float] = []

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

            if nlp_scispacy is not None:
                chem, dis, overall = compute_ner_f1_by_type(nlp_scispacy, ref, hyp)
                row["chemical_f1"] = round(chem, 4)
                row["disease_f1"] = round(dis, 4)
                row["scispacy_overall_f1"] = round(overall, 4)
                chemical_vals.append(chem)
                disease_vals.append(dis)
                overall_vals.append(overall)

            rows.append(row)
            print(f"  {sample.get('speaker_id', '')}: WER={m['wer']:.3f} BLEU={m['bleu']:.3f}")

        if rows:
            out_path = METRICS_DIR / f"{cond}_metrics.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
            print(f"Saved {out_path} (n={len(rows)})")

            avg_wer = np.mean([r["wer"] for r in rows])
            summary[cond] = {
                "wer": float(avg_wer),
                "bleu": float(np.mean([r["bleu"] for r in rows])),
                "rouge_l": float(np.mean([r["rouge_l"] for r in rows])),
                "bert_score": float(np.mean([r["bert_score"] for r in rows])),
            }
            if nlp_scispacy is not None and chemical_vals:
                summary[cond]["chemical_f1"] = float(np.mean(chemical_vals))
                summary[cond]["disease_f1"] = float(np.mean(disease_vals))
                summary[cond]["scispacy_overall_f1"] = float(np.mean(overall_vals))
                print(f"  CHEMICAL F1: {summary[cond]['chemical_f1']:.4f}  DISEASE F1: {summary[cond]['disease_f1']:.4f}")
            print(f"  Mean WER: {avg_wer:.4f}\n")

            # Per-dialogue CSV
            csv_path = PER_DIALOGUE_DIR / f"primock57_per_dialogue_{cond}.csv"
            cols = ["sample_id", "speaker_id", "WER", "BLEU", "ROUGE_L", "BERTScore"]
            if nlp_scispacy is not None:
                cols.extend(["CHEMICAL_F1", "DISEASE_F1", "SCISPACY_OVERALL_F1"])
            with csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                for r in rows:
                    rw = [r.get("sample_id"), r.get("speaker_id"), r.get("wer"), r.get("bleu"), r.get("rouge_l"), r.get("bert_score")]
                    if nlp_scispacy is not None:
                        rw.extend([
                            r.get("chemical_f1", ""),
                            r.get("disease_f1", ""),
                            r.get("scispacy_overall_f1", ""),
                        ])
                    w.writerow(rw)
            print(f"  Per-dialogue: {csv_path}")

    if summary:
        sum_path = METRICS_DIR / "primock57_metrics_summary.json"
        with sum_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary: {sum_path}")

    print("PriMock57 speech eval (with NER) done.")


if __name__ == "__main__":
    main()
