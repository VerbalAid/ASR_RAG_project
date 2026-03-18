#!/usr/bin/env bash
# PriMock57 speech evaluation with scispaCy NER (CHEMICAL, DISEASE F1).
# Enables real-vs-simulated comparison with MTS-Dialog.
# From project root:  bash scripts/run_primock57_eval_with_ner.sh

set -e
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
fi

echo "=== PriMock57 speech eval (WER, BLEU, ROUGE-L, BERTScore + scispaCy CHEMICAL/DISEASE F1) ==="
"$PYTHON" -m analysis.primock57_speech_eval_with_ner

echo ""
echo "=== PriMock57 speech figure (incl. scispaCy if available) ==="
"$PYTHON" visualisations/primock57_speech_summary.py

echo ""
echo "Done. Metrics: results/primock57_speech/metrics/"
echo "Figure: images/primock57_speech_cond_metrics.png"
