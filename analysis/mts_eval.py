import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from bert_score import score as bert_score
from jiwer import wer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


RESULTS_DIR = Path("results") / "mts"
METRICS_DIR = RESULTS_DIR / "metrics"
PER_DIALOGUE_DIR = RESULTS_DIR / "per_dialogue"
SCISPACY_DIR = RESULTS_DIR / "scispacy"
BERT_GPU_BATCH_SIZE = 8


def extract_ollama_content(raw: str) -> str:
    """Extract assistant message content from Ollama response string."""
    if not raw or not isinstance(raw, str):
        return ""
    m = re.search(r'content="((?:[^"\\]|\\.)*)"\s*,\s*thinking', raw, re.DOTALL)
    if m:
        return m.group(1).replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\").strip()
    m2 = re.search(r"content='((?:[^'\\]|\\.)*)'\s*,\s*thinking", raw, re.DOTALL)
    if m2:
        return m2.group(1).replace("\\'", "'").replace("\\n", "\n").strip()
    return ""


def pick_bert_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def compute_bert_f1(
    hyps: List[str], refs: List[str], preferred_device: str
) -> List[float]:
    """Compute BERTScore F1 with GPU-first strategy and CPU fallback on OOM."""
    try:
        _, _, f1 = bert_score(
            hyps,
            refs,
            lang="en",
            verbose=False,
            device=preferred_device,
            batch_size=BERT_GPU_BATCH_SIZE if preferred_device.startswith("cuda") else 64,
        )
        return [float(x) for x in f1.tolist()]
    except RuntimeError as e:
        # Keep run robust on small VRAM GPUs.
        if preferred_device.startswith("cuda") and "out of memory" in str(e).lower():
            print("[mts_eval] CUDA OOM in BERTScore; retrying on CPU.")
            _, _, f1 = bert_score(hyps, refs, lang="en", verbose=False, device="cpu")
            return [float(x) for x in f1.tolist()]
        raise


def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Metrics:
    wer: float
    bleu: float
    rouge_l: float
    bert: float
    ner_f1: float
    chemical_f1: Optional[float] = None
    disease_f1: Optional[float] = None
    scispacy_overall_f1: Optional[float] = None


def compute_text_metrics(
    ref: str, hyp: str, *, smoothing_fn, rouge_scorer
) -> Tuple[float, float, float]:
    ref_n = normalise(ref)
    hyp_n = normalise(hyp)
    if not hyp_n:
        return 1.0, 0.0, 0.0
    w = wer(ref_n, hyp_n)
    b = sentence_bleu([ref_n.split()], hyp_n.split(), smoothing_function=smoothing_fn)
    r = rouge_scorer.score(ref_n, hyp_n)["rougeL"].fmeasure
    return w, b, r


def compute_ner_f1(nlp, ref: str, hyp: str) -> float:
    doc_ref = nlp(ref)
    doc_hyp = nlp(hyp)
    ref_ents = {ent.text.lower() for ent in doc_ref.ents}
    hyp_ents = {ent.text.lower() for ent in doc_hyp.ents}
    if not ref_ents and not hyp_ents:
        return 1.0
    if not ref_ents or not hyp_ents:
        return 0.0
    tp = len(ref_ents & hyp_ents)
    fp = len(hyp_ents - ref_ents)
    fn = len(ref_ents - hyp_ents)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_ner_f1_by_type(
    nlp, ref: str, hyp: str
) -> Tuple[float, float, float]:
    """
    Compute NER F1 per entity type (for scispaCy CHEMICAL, DISEASE) and overall.
    Returns (chemical_f1, disease_f1, overall_f1).
    """
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


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PER_DIALOGUE_DIR.mkdir(parents=True, exist_ok=True)
    SCISPACY_DIR.mkdir(parents=True, exist_ok=True)

    cond_to_field: Dict[str, str] = {
        "c1": "noisy",
        "c2a": "c2a_llama",
        "c2b": "c2b_mistral",
        "c3_lex_gen": "c3_bm25_rag",
        "c3_lex_rel": "c3_bm25_rag",
        "c3_lex_mat": "c3_bm25_rag",
        "c4_den_gen": "c4a_dense_generic",
        "c4_den_rel": "c4a_dense_generic",
        "c4_den_mat": "c4b_dense_clinical",
    }

    smoothing_fn = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bert_device = pick_bert_device()
    print(f"[mts_eval] BERTScore device: {bert_device}")

    nlp: Any = None
    nlp_scispacy: Any = None
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("Loaded spaCy (en_core_web_sm) for NER F1.")
    except Exception as e:
        print(f"[mts_eval] spaCy en_core_web_sm not available: ({e})")
        print("[mts_eval] To enable NER: pip install spacy && python -m spacy download en_core_web_sm")
    try:
        import spacy
        nlp_scispacy = spacy.load("en_ner_bc5cdr_md")
        print("Loaded scispaCy (en_ner_bc5cdr_md) for CHEMICAL/DISEASE F1.")
        if nlp is None:
            print("[mts_eval] Using scispaCy overall F1 as NER proxy (en_core_web_sm unavailable).")
    except OSError:
        if nlp is None:
            print("[mts_eval] scispaCy en_ner_bc5cdr_md not found; NER omitted. Install: pip install scispacy && pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz")

    metrics_by_cond: Dict[str, List[Metrics]] = {c: [] for c in cond_to_field}

    for cond, field in cond_to_field.items():
        path = RESULTS_DIR / {
            "c3_lex_gen": "mts_c3_lex_gen_results.json",
            "c3_lex_rel": "mts_c3_lex_rel_results.json",
            "c3_lex_mat": "mts_c3_lex_mat_results.json",
            "c4_den_rel": "mts_c4_den_rel_results.json",
        }.get(cond, f"mts_{cond}_results.json")
        if not path.exists():
            print(f"[mts_eval] Skipping {cond} – {path} not found.")
            continue

        print(f"[mts_eval] Evaluating condition {cond} from {path}...")
        with path.open("r") as f:
            records = json.load(f)

        c4_den_mat_cleaned: Optional[List[str]] = None
        if cond == "c4_den_mat":
            cleaned_path = RESULTS_DIR / "cleaned" / "c4_den_mat_cleaned.json"
            if cleaned_path.exists():
                with cleaned_path.open("r") as f:
                    c4_den_mat_cleaned = json.load(f)
                print(f"[mts_eval] Using {len(c4_den_mat_cleaned)} cleaned C4-Den-Mat hypotheses.")

        refs_for_bert: List[str] = []
        hyps_for_bert: List[str] = []
        text_metric_rows: List[Tuple[float, float, float]] = []
        ner_rows: List[float] = []
        scispacy_rows: List[Tuple[float, float, float]] = []
        dialogue_ids: List[int] = []

        for i, rec in enumerate(records):
            ref = rec.get("dialogue") or rec.get("ground_truth") or ""
            dialogue_id = int(rec.get("sample_id", i + 1))

            if c4_den_mat_cleaned is not None and i < len(c4_den_mat_cleaned):
                hyp = c4_den_mat_cleaned[i]
            else:
                raw = rec.get(field) or ""
                if cond == "c1":
                    hyp = raw
                else:
                    hyp = extract_ollama_content(raw) or raw
            if not ref:
                continue

            w, b, r = compute_text_metrics(
                ref, hyp, smoothing_fn=smoothing_fn, rouge_scorer=rouge
            )
            text_metric_rows.append((w, b, r))
            if nlp_scispacy is not None:
                chem, dis, overall = compute_ner_f1_by_type(nlp_scispacy, ref, hyp)
                scispacy_rows.append((chem, dis, overall))
                if nlp is not None:
                    ner_rows.append(compute_ner_f1(nlp, ref, hyp))
                else:
                    ner_rows.append(float(overall) if overall is not None else math.nan)
            else:
                scispacy_rows.append((None, None, None))
                if nlp is not None:
                    ner_rows.append(compute_ner_f1(nlp, ref, hyp))
                else:
                    ner_rows.append(math.nan)
            refs_for_bert.append(normalise(ref))
            hyps_for_bert.append(normalise(hyp))
            dialogue_ids.append(dialogue_id)

        if refs_for_bert:
            bert_rows = compute_bert_f1(hyps_for_bert, refs_for_bert, bert_device)
        else:
            bert_rows = []

        for j, ((w, b, r), bs, ner_f) in enumerate(zip(text_metric_rows, bert_rows, ner_rows)):
            sc = scispacy_rows[j] if j < len(scispacy_rows) else (None, None, None)
            m = Metrics(
                w, b, r, bs, ner_f,
                chemical_f1=sc[0], disease_f1=sc[1], scispacy_overall_f1=sc[2],
            )
            metrics_by_cond[cond].append(m)

        # Per-dialogue CSV for this condition
        _csv_suffix = cond
        csv_path = PER_DIALOGUE_DIR / f"mts_per_dialogue_{_csv_suffix}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dialogue_id", "WER", "BLEU", "ROUGE_L", "BERTScore", "NER_F1",
                "CHEMICAL_F1", "DISEASE_F1", "SCISPACY_OVERALL_F1",
            ])
            for k, m in enumerate(metrics_by_cond[cond]):
                did = dialogue_ids[k] if k < len(dialogue_ids) else k + 1
                ner_val = m.ner_f1 if not math.isnan(m.ner_f1) else ""
                row = [did, m.wer, m.bleu, m.rouge_l, m.bert, ner_val]
                row.extend([
                    m.chemical_f1 if m.chemical_f1 is not None else "",
                    m.disease_f1 if m.disease_f1 is not None else "",
                    m.scispacy_overall_f1 if m.scispacy_overall_f1 is not None else "",
                ])
                writer.writerow(row)
        print(f"[mts_eval] Wrote {csv_path} ({len(metrics_by_cond[cond])} rows).")

        # ScispaCy-only CSV in results/mts/scispacy/
        if nlp_scispacy is not None and metrics_by_cond[cond]:
            scispacy_path = SCISPACY_DIR / f"mts_per_dialogue_scispacy_{_csv_suffix}.csv"
            with scispacy_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["dialogue_id", "CHEMICAL_F1", "DISEASE_F1", "SCISPACY_OVERALL_F1"])
                for k, m in enumerate(metrics_by_cond[cond]):
                    did = dialogue_ids[k] if k < len(dialogue_ids) else k + 1
                    writer.writerow([
                        did,
                        m.chemical_f1 if m.chemical_f1 is not None else "",
                        m.disease_f1 if m.disease_f1 is not None else "",
                        m.scispacy_overall_f1 if m.scispacy_overall_f1 is not None else "",
                    ])

    summary: Dict[str, Dict[str, float]] = {}
    for cond, vals in metrics_by_cond.items():
        if not vals:
            continue
        ner_vals = [v.ner_f1 for v in vals]
        ner_mean = float(np.nanmean(ner_vals)) if any(not math.isnan(x) for x in ner_vals) else math.nan
        summary[cond] = {
            "wer": float(np.mean([v.wer for v in vals])),
            "bleu": float(np.mean([v.bleu for v in vals])),
            "rouge_l": float(np.mean([v.rouge_l for v in vals])),
            "bert": float(np.mean([v.bert for v in vals])),
            "ner_f1": ner_mean,
        }
        if vals[0].chemical_f1 is not None:
            summary[cond]["chemical_f1"] = float(np.nanmean([v.chemical_f1 if v.chemical_f1 is not None else np.nan for v in vals]))
            summary[cond]["disease_f1"] = float(np.nanmean([v.disease_f1 if v.disease_f1 is not None else np.nan for v in vals]))
            summary[cond]["scispacy_overall_f1"] = float(np.nanmean([v.scispacy_overall_f1 if v.scispacy_overall_f1 is not None else np.nan for v in vals]))

    # Add C4-Den-Mat WER note if re-run with verbosity fix (optional)
    if "c4_den_mat" in summary:
        summary["c4_den_mat_wer_note"] = "inflated by verbosity, not comparable"

    # C2a: reported in appendix only (confounded vs C2b)
    summary["c2a_note"] = "confounded by simultaneous model and prompt change vs C2b; reported in appendix only"

    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    out_path = METRICS_DIR / "mts_metrics_summary.json"
    with out_path.open("w") as f:
        json.dump(_sanitize(summary), f, indent=2)

    print("\n=== MTS-Dialog metrics summary ===")
    for cond, m in summary.items():
        if not isinstance(m, dict) or "wer" not in m:
            continue
        ner_str = f"{m['ner_f1']:.4f}" if not math.isnan(m["ner_f1"]) else "n/a"
        line = f"{cond}: WER={m['wer']:.4f}, BLEU={m['bleu']:.4f}, ROUGE-L={m['rouge_l']:.4f}, BERT={m['bert']:.4f}, NER-F1={ner_str}"
        if "chemical_f1" in m:
            line += f", CHEMICAL_F1={m['chemical_f1']:.4f}, DISEASE_F1={m['disease_f1']:.4f}"
        print(line)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()

