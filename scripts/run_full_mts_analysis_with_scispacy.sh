#!/usr/bin/env bash
# Full MTS analysis: install scispaCy (if needed), run mts_eval with CHEMICAL/DISEASE F1,
# Wilcoxon tests, sanity check, and visualisations.
# From project root:  bash scripts/run_full_mts_analysis_with_scispacy.sh

set -e
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
fi

echo "=== 1. Install scispaCy and en_ner_bc5cdr_md (if not present) ==="
$PYTHON -c "
import spacy
try:
    spacy.load('en_ner_bc5cdr_md')
    print('en_ner_bc5cdr_md already installed.')
except OSError:
    print('Installing scispaCy model...')
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
        'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz'])
" 2>/dev/null || {
  echo "Installing scispaCy model (may take a few minutes)..."
  $PYTHON -m pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz" || true
}

echo ""
echo "=== 2. MTS evaluation (with scispaCy CHEMICAL/DISEASE F1 if installed) ==="
$PYTHON -m analysis.mts_eval

echo ""
echo "=== 3. Wilcoxon statistical tests ==="
$PYTHON -m analysis.wilcoxon_mts

echo ""
echo "=== 4. Sanity check ==="
$PYTHON -m analysis.sanity_check

echo ""
echo "=== 5. Visualisations ==="
$PYTHON visualisations/mts_summary.py
$PYTHON visualisations/ner_bert_divergence.py
$PYTHON visualisations/ner_comparison.py

echo ""
echo "Done. MTS metrics: results/mts/metrics/"
echo "ScispaCy per-dialogue CSVs: results/mts/per_dialogue/ (CHEMICAL_F1, DISEASE_F1 columns)"
