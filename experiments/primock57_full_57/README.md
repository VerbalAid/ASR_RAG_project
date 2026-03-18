# PriMock57 Full (57 consultations): Definitive C2b vs C4-Den-Mat

This folder runs **only** the conditions needed for the statistically definitive claim on real clinical audio:

- **C1** (Whisper-tiny baseline)
- **C2b** (Mistral-only correction)
- **C4-Den-Mat** (clinical dense RAG over MTS-Dialog)

Optional: **C4-Den-Gen** (generic dense) for a three-way comparison.

**Goal:** *"On real clinical audio (n=57), clinical dense RAG significantly underperforms LLM-only correction (p < 0.01, Wilcoxon), confirming that phonetically motivated errors cannot be recovered from text context alone."*

## Prerequisites

- **Python env:** Activate the project venv and install dependencies so `soundfile` and `whisper` are available:  
  `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows), then `pip install -r requirements.txt`.
- `analysis/prepare_primock57.py` run once (produces `results/primock57/primock57_passages.json`; needed for consistency; MTS passages are loaded by run script).
- **Audio for all 57 consultations** in `datasets/primock57_raw/audio_consultations/` as `{consultation_id}_doctor.wav` and `{consultation_id}_patient.wav`. The repo stores these with **Git LFS**. To download them:

  **Option A (recommended):** Run the project script (requires **git** and **git-lfs**):

  ```bash
  # Install git-lfs if needed (once per machine):
  #   Fedora:  dnf install git-lfs && git lfs install
  #   macOS:   brew install git-lfs && git lfs install
  python scripts/download_primock57_audio.py
  ```

  This clones [babylonhealth/primock57](https://github.com/babylonhealth/primock57), runs `git lfs pull`, and copies all WAVs into `datasets/primock57_raw/audio_consultations/`. Use `--no-keep` to delete the staging clone after copying.

  **Option B:** Clone the repo yourself with LFS and copy the `audio/` folder contents into `datasets/primock57_raw/audio_consultations/` (same filenames: `day1_consultation01_doctor.wav`, etc.).

## Steps (from project root)

### 0. Download audio (if not already present)

```bash
python scripts/download_primock57_audio.py
```

Requires **git** and **git-lfs** (see Prerequisites). If you already have WAVs in `datasets/primock57_raw/audio_consultations/`, skip this.

### 1. Transcribe all 57 (C1)

```bash
python -m experiments.primock57_full_57.c1_whisper_57
```

- Reads ground truth from TextGrids, concatenates doctor+patient WAV, runs Whisper-tiny.
- Writes `results/primock57_full_57/c1_whisper_raw.json` (list of `{sample_id, speaker_id, ground_truth, c1_whisper_tiny}`).
- **Time:** ~6 min per consultation → ~5.7 hours for 57 (Whisper on CPU/GPU).

### 2. Run C2b and C4-Den-Mat (and optionally C4-Den-Gen)

```bash
# Core: C2b + C4-Den-Mat only (~11–12 hours for 57)
python -m experiments.primock57_full_57.run_c2b_c4

# With C4-Den-Gen as well (~17 hours total)
python -m experiments.primock57_full_57.run_c2b_c4 --add-c4-den-gen
```

- Loads `results/primock57_full_57/c1_whisper_raw.json`.
- Runs Mistral-only (C2b) and dense MTS-Dialog clinical (C4-Den-Mat) per consultation; optionally C4-Den-Gen (Wikipedia).
- Writes `c2b_outputs.json`, `c4_den_mat_outputs.json`, and (if requested) `c4_den_gen_outputs.json` into `results/primock57_full_57/`.
- **Time:** ~6 min per consultation per condition → 57 × 2 × ~6 min ≈ **11.4 h** (C2b + C4-Den-Mat); with C4-Den-Gen add ~5.7 h.

### 3. Evaluate and run Wilcoxon (n=57)

```bash
python -m experiments.primock57_full_57.eval_57
```

- Computes WER, BLEU, ROUGE-L, BERTScore per sample for C1, C2b, C4-Den-Mat (and C4-Den-Gen if present).
- Writes `results/primock57_full_57/per_dialogue/*.csv` and `results/primock57_full_57/metrics/primock57_full_57_summary.json`.
- Runs paired Wilcoxon (C4-Den-Mat vs C2b; optionally C4-Den-Gen vs C2b) and prints/saves results to `results/primock57_full_57/metrics/primock57_full_57_wilcoxon.md`.

## Outputs

| Path | Description |
|------|-------------|
| `results/primock57_full_57/c1_whisper_raw.json` | C1 transcriptions (57 samples) |
| `results/primock57_full_57/c2b_outputs.json` | C2b (Mistral-only) outputs |
| `results/primock57_full_57/c4_den_mat_outputs.json` | C4-Den-Mat (clinical dense) outputs |
| `results/primock57_full_57/per_dialogue/*.csv` | Per-sample metrics |
| `results/primock57_full_57/metrics/primock57_full_57_summary.json` | Mean metrics per condition |
| `results/primock57_full_57/metrics/primock57_full_57_wilcoxon.md` | Wilcoxon C4-Den-Mat vs C2b (and optional C4-Den-Gen vs C2b) |

## Resumability

- **C1:** Overwrites `c1_whisper_raw.json` each run. Re-run to regenerate.
- **run_c2b_c4:** Can be extended later to support `--resume` (not implemented yet); for now run to completion or re-run from scratch.

## Relation to main pipeline

- **Main pipeline:** `experiments/primock57_speech/` uses the **first 10** consultations and runs all conditions (C2a, C2b, C3*, C4*).
- **This pipeline:** Uses **all 57** consultations and only C2b + C4-Den-Mat (and optionally C4-Den-Gen) for a clean, publishable comparison on real clinical audio.
