"""Remove DanBarber_2010_S103 from TED metrics and output JSONs."""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "ted"
METRICS_DIR = RESULTS_DIR / "metrics"
EXCLUDE_SPEAKER = "DanBarber_2010_S103"

METRICS_FILES = [
    "c1_metrics.json",
    "c2_metrics_llama.json",
    "c2_metrics_mistral.json",
    "c3_lex_gen_metrics.json",
    "c3_lex_rel_metrics.json",
    "c3_lex_mat_metrics.json",
    "c4_den_gen_metrics.json",
    "c4_den_mat_metrics.json",
]

OUTPUT_FILES = [
    "c1_whisper_raw.json",
    "c2_outputs_llama.json",
    "c2_outputs_mistral.json",
    "c3_lex_gen_outputs_bm25.json",
    "ted_c3_lex_rel_results.json",
    "ted_c3_lex_mat_results.json",
    "c4_den_gen_outputs.json",
    "c4_den_mat_outputs.json",
]


def main():
    # Remove from metrics and renumber sample_id
    for name in METRICS_FILES:
        path = METRICS_DIR / name
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        new_data = [r for r in data if r.get("speaker_id") != EXCLUDE_SPEAKER]
        for i, r in enumerate(new_data):
            r["sample_id"] = i
        with path.open("w") as f:
            json.dump(new_data, f, indent=2)
        print(f"  {name}: {len(data)} -> {len(new_data)} rows")

    # Remove from output JSONs (list of records with speaker_id)
    for name in OUTPUT_FILES:
        path = RESULTS_DIR / name
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        if isinstance(data, list):
            new_data = [r for r in data if r.get("speaker_id") != EXCLUDE_SPEAKER]
        else:
            new_data = data
        with path.open("w") as f:
            json.dump(new_data, f, indent=2)
        if isinstance(data, list):
            print(f"  {name}: {len(data)} -> {len(new_data)} rows")
        else:
            print(f"  {name}: (structure preserved)")

    print(f"\nRemoved speaker: {EXCLUDE_SPEAKER}. TED now has 10 speakers.")


if __name__ == "__main__":
    main()
