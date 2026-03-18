#!/usr/bin/env bash
# Run all visualisation scripts. Use after eval steps so figures reflect current data.
# From project root:  bash scripts/run_visualisations.sh

set -e
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
fi

echo "=== TED figures ==="
"$PYTHON" visualisations/ted_summary.py
"$PYTHON" visualisations/ted_wer_heat.py
"$PYTHON" visualisations/improvement_chart.py
"$PYTHON" visualisations/retrieval_luck.py
echo ""
echo "=== MTS figures (includes scispaCy CHEMICAL/DISEASE if mts_eval was run with scispaCy) ==="
"$PYTHON" visualisations/mts_summary.py
"$PYTHON" visualisations/ner_bert_divergence.py
echo ""
echo "=== PriMock57 speech figure ==="
"$PYTHON" visualisations/primock57_speech_summary.py
echo ""
echo "Done. Outputs in images/"
