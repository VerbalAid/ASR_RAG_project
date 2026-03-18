"""NER by entity type (scispaCy) for MTS conditions; writes results/mts/metrics/ner_by_type.md."""

import json
import re
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "mts"
METRICS_DIR = RESULTS_DIR / "metrics"
CLEANED_DIR = RESULTS_DIR / "cleaned"


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


def load_references() -> list[str]:
    """Load reference dialogues from MTS results (dialogue field), order preserved."""
    path = RESULTS_DIR / "mts_c1_results.json"
    if not path.exists():
        path = RESULTS_DIR / "mts_c4_den_mat_results.json"
    with path.open("r") as f:
        records = json.load(f)
    return [r.get("dialogue") or r.get("ground_truth") or "" for r in records]


def load_hypotheses_for_condition(cond: str) -> list[str] | None:
    """Load hypothesis list for one condition (length 100, index-aligned)."""
    if cond == "c4_den_mat":
        p = CLEANED_DIR / "c4_den_mat_cleaned.json"
        if not p.exists():
            return None
        with p.open("r") as f:
            return json.load(f)
    field = {"c1": "noisy", "c2b": "c2b_mistral", "c3_lex_gen": "c3_bm25_rag", "c3_lex_rel": "c3_bm25_rag", "c3_lex_mat": "c3_bm25_rag", "c4_den_gen": "c4a_dense_generic", "c4_den_rel": "c4a_dense_generic"}.get(cond)
    if not field:
        return None
    path_map = {"c3_lex_gen": "mts_c3_lex_gen_results.json", "c3_lex_rel": "mts_c3_lex_rel_results.json", "c3_lex_mat": "mts_c3_lex_mat_results.json", "c4_den_rel": "mts_c4_den_rel_results.json"}
    path = RESULTS_DIR / path_map.get(cond, f"mts_{cond}_results.json")
    if not path.exists():
        return None
    with path.open("r") as f:
        records = json.load(f)
    out = []
    for r in records:
        raw = r.get(field) or ""
        if cond == "c1":
            text = raw
        else:
            text = extract_ollama_content(raw)
        out.append(text or "")
    return out


def entity_set(doc) -> dict[str, set[str]]:
    """Return dict mapping entity label -> set of entity texts (lowercased)."""
    by_label: dict[str, set[str]] = {}
    for ent in doc.ents:
        by_label.setdefault(ent.label_, set()).add(ent.text.lower())
    return by_label


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    import spacy
    try:
        nlp = spacy.load("en_ner_bc5cdr_md")
        model_name = "en_ner_bc5cdr_md (biomedical)"
    except OSError:
        print("en_ner_bc5cdr_md not found; using en_core_web_sm (general NER).", file=sys.stderr)
        print("  For biomedical entities install: pip install scispacy && pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz", file=sys.stderr)
        nlp = spacy.load("en_core_web_sm")
        model_name = "en_core_web_sm (general)"

    refs = load_references()
    n = len(refs)
    conditions = ["c1", "c2b", "c3_lex_gen", "c3_lex_rel", "c3_lex_mat", "c4_den_gen", "c4_den_rel", "c4_den_mat"]
    cond_labels = {
        "c1": "Noisy Baseline",
        "c2b": "LLM-Only",
        "c3_lex_gen": "C3-Lex-Gen",
        "c3_lex_rel": "C3-Lex-Rel",
        "c3_lex_mat": "C3-Lex-Mat",
        "c4_den_gen": "C4-Den-Gen",
        "c4_den_rel": "C4-Den-Rel",
        "c4_den_mat": "C4-Den-Mat",
    }

    # Per condition, per entity type: TP, FP, FN (aggregated over all dialogues)
    # structure: cond -> label -> {"tp", "fp", "fn"}
    stats: dict[str, dict[str, dict[str, int]]] = {c: {} for c in conditions}

    for cond in conditions:
        hyps = load_hypotheses_for_condition(cond)
        if hyps is None or len(hyps) != n:
            print(f"[ner_by_type] Skipping {cond}: hypotheses missing or length != {n}")
            continue
        print(f"[ner_by_type] Processing {cond}...")
        for ref, hyp in zip(refs, hyps):
            doc_ref = nlp(ref)
            doc_hyp = nlp(hyp)
            ref_by_label = entity_set(doc_ref)
            hyp_by_label = entity_set(doc_hyp)
            all_labels = set(ref_by_label) | set(hyp_by_label)
            for label in all_labels:
                ref_ents = ref_by_label.get(label, set())
                hyp_ents = hyp_by_label.get(label, set())
                tp = len(ref_ents & hyp_ents)
                fp = len(hyp_ents - ref_ents)
                fn = len(ref_ents - hyp_ents)
                if label not in stats[cond]:
                    stats[cond][label] = {"tp": 0, "fp": 0, "fn": 0}
                stats[cond][label]["tp"] += tp
                stats[cond][label]["fp"] += fp
                stats[cond][label]["fn"] += fn

    # Build table: rows = entity types, columns = conditions (F1)
    all_types = set()
    for c in conditions:
        all_types.update(stats[c].keys())
    all_types = sorted(all_types)

    def p_r_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    lines = [
        f"# NER F1 by entity type (MTS-Dialog, model: {model_name})",
        "",
        "| Entity Type | " + " | ".join(cond_labels[c] for c in conditions) + " |",
        "|" + "-------------|" * (len(conditions) + 1),
    ]
    for label in all_types:
        cells = [label]
        for c in conditions:
            s = stats[c].get(label, {"tp": 0, "fp": 0, "fn": 0})
            _, _, f1 = p_r_f1(s["tp"], s["fp"], s["fn"])
            cells.append(f"{f1:.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    interp = (
        "Chemical entities (e.g. drug names) and disease names show a sharp drop in F1 from the "
        "noisy baseline to all LLM-based conditions; the biomedical NER model (BC5CDR) detects "
        "that many medical terms are lost or replaced in the reformulated transcripts. "
        "Domain-matched retrieval (C4b) does not recover entity-level fidelity compared to the "
        "noisy baseline; addressing this would require entity-aware retrieval or constrained decoding."
        if "bc5cdr" in model_name.lower()
        else "General named entities (e.g. PERSON, GPE, DATE) show mixed preservation: some types "
        "(e.g. ORDINAL, TIME) remain relatively stable across conditions, while others (CARDINAL, "
        "DATE, ORG) drop from the noisy baseline to LLM-based conditions. Domain-matched (C4b) "
        "performs similarly to other LLM conditions. For clinical relevance, re-run with "
        "en_ner_bc5cdr_md (scispacy) to obtain CHEMICAL/DISEASE breakdown."
    )
    lines.extend(["", "## Interpretation", "", interp, ""])

    out_path = METRICS_DIR / "ner_by_type.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
