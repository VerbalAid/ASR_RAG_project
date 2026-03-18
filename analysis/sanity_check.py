"""Print TED and MTS condition grids (2×3) and C1 WER for quick verification."""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
TED_METRICS = RESULTS / "ted" / "metrics"
MTS_METRICS = RESULTS / "mts" / "metrics"
MTS_SUMMARY = MTS_METRICS / "mts_metrics_summary.json"


TED_GRID = [
    ("BM25", "Generic", "c3_lex_gen"),
    ("BM25", "Domain-relevant", "c3_lex_rel"),
    ("BM25", "Domain-matched", "c3_lex_mat"),
    ("Dense", "Generic", "c4_den_gen"),
    ("Dense", "Domain-relevant", None),  # TED has no C4-Den-Rel
    ("Dense", "Domain-matched", "c4_den_mat"),
]

MTS_GRID = [
    ("BM25", "Generic", "c3_lex_gen"),
    ("BM25", "Domain-relevant", "c3_lex_rel"),
    ("BM25", "Domain-matched", "c3_lex_mat"),
    ("Dense", "Generic", "c4_den_gen"),
    ("Dense", "Domain-relevant", "c4_den_rel"),
    ("Dense", "Domain-matched", "c4_den_mat"),
]


def mean_wer_from_metrics(path: Path) -> float:
    if not path.exists():
        return float("nan")
    with path.open() as f:
        data = json.load(f)
    if not data:
        return float("nan")
    wers = [r["wer"] for r in data if "wer" in r]
    return sum(wers) / len(wers) if wers else float("nan")


def main():
    print("=" * 70)
    print("Condition grid sanity check")
    print("=" * 70)

    # TED
    print("\n--- TED-LIUM (n=10) ---")
    print(f"{'Modality':<10} {'Corpus':<20} {'Condition':<12} {'WER':<8} {'n'}")
    print("-" * 65)
    ted_n = 10
    for mod, corp, cond in TED_GRID:
        if cond is None:
            print(f"{mod:<10} {corp:<20} {'—':<12} {'—':<8} —")
            continue
        metrics_file = TED_METRICS / f"{cond}_metrics.json"
        wer = mean_wer_from_metrics(metrics_file)
        n = ted_n if metrics_file.exists() else "?"
        print(f"{mod:<10} {corp:<20} {cond:<12} {wer:<8.4f} {n}")

    # MTS
    print("\n--- MTS-Dialog (n=100) ---")
    print(f"{'Modality':<10} {'Corpus':<20} {'Condition':<12} {'WER':<8} {'n'}")
    print("-" * 65)
    mts_summary = {}
    if MTS_SUMMARY.exists():
        with MTS_SUMMARY.open() as f:
            mts_summary = json.load(f)
    mts_n = 100
    for mod, corp, cond in MTS_GRID:
        if cond is None:
            continue
        wer = mts_summary.get(cond, {}).get("wer", float("nan"))
        if isinstance(wer, dict):
            wer = wer.get("wer", float("nan"))
        n = mts_n if cond in mts_summary else "?"
        note = ""
        if cond == "c4_den_mat" and mts_summary.get("c4_den_mat_wer_note"):
            note = "  ← " + mts_summary.get("c4_den_mat_wer_note")
        print(f"{mod:<10} {corp:<20} {cond:<12} {wer:<8.4f} {n}{note}")

    print("\n--- C1 baseline ---")
    ted_c1 = mean_wer_from_metrics(TED_METRICS / "c1_metrics.json")
    mts_c1 = mts_summary.get("c1", {}).get("wer", float("nan"))
    print(f"TED C1 WER: {ted_c1:.4f}  MTS C1 WER: {mts_c1:.4f}")

    print("\n--- 2×3 completeness ---")
    ted_missing = [c for (mod, corp, c) in TED_GRID if c is not None and not (TED_METRICS / f"{c}_metrics.json").exists()]
    mts_missing = [c for (mod, corp, c) in MTS_GRID if c is not None and (c not in mts_summary or not isinstance(mts_summary.get(c), dict) or "wer" not in mts_summary.get(c, {}))]
    if ted_missing:
        print(f"TED missing: {ted_missing}")
    else:
        print("TED: 2×3 grid complete")
    if mts_missing:
        print(f"MTS missing: {mts_missing}")
    else:
        print("MTS: 2×3 grid complete")
    print()


if __name__ == "__main__":
    main()
