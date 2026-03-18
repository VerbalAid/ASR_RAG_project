#!/usr/bin/env bash
# Full pipeline: TED + MTS + PriMock57. Run from project root: bash run_all.sh

set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
fi

echo "Warming up Ollama (mistral:latest)..."
ollama run mistral:latest "Say OK" >/dev/null 2>&1 || true

echo "=== Step 1 — MTS C4-Den-Gen ==="
"$PYTHON" -m experiments.mts.run c4a --c4a-corpus ag_news
echo ""

echo "=== Step 2 — BM25 conditions (TED + MTS) ==="
"$PYTHON" experiments/test/c3_bm25.py
"$PYTHON" -m experiments.mts.run c3 --c3-corpus ag_news
"$PYTHON" experiments/test/c3b_ted_bm25.py
"$PYTHON" -m experiments.mts.run c3 --c3-corpus primock57
"$PYTHON" experiments/test/ted_c3_lex_mat.py
"$PYTHON" -m experiments.mts.run c3_mat
echo ""

echo "=== Step 3 — TED evaluations ==="
"$PYTHON" analysis/c3_bm25_eval.py
"$PYTHON" analysis/ted_c3b_eval.py
"$PYTHON" analysis/ted_c3_lex_mat_eval.py
echo ""

echo "=== Step 4 — MTS evaluation ==="
"$PYTHON" -m analysis.mts_eval
echo ""

echo "=== Step 5 — Sanity check ==="
"$PYTHON" -m analysis.sanity_check
echo ""

echo "=== Step 6 — Wilcoxon ==="
"$PYTHON" -m analysis.statistical_significance
"$PYTHON" -m analysis.wilcoxon_mts
echo ""

echo "=== Step 7 — Figures (TED + MTS) ==="
"$PYTHON" visualisations/ted_summary.py
"$PYTHON" visualisations/ted_wer_heat.py
"$PYTHON" visualisations/improvement_chart.py
"$PYTHON" visualisations/retrieval_luck.py
"$PYTHON" visualisations/mts_summary.py
"$PYTHON" visualisations/ner_bert_divergence.py
echo ""

echo "=== Step 8 — PriMock57 prepare ==="
"$PYTHON" -m analysis.prepare_primock57
echo ""

echo "=== Step 9 — PriMock57 C1 (Whisper, 10 consultations) ==="
"$PYTHON" -m experiments.primock57_speech.c1_whisper
echo ""

echo "=== Step 10 — PriMock57 RAG ==="
"$PYTHON" -m experiments.primock57_speech.run_rag
echo ""

echo "=== Step 11 — PriMock57 eval ==="
"$PYTHON" -m analysis.primock57_speech_eval
echo ""

echo "=== Step 12 — PriMock57 figure ==="
"$PYTHON" visualisations/primock57_speech_summary.py
echo ""

echo "=== Final sanity check ==="
"$PYTHON" -m analysis.sanity_check
echo "Done. Metrics: results/{ted,mts,primock57_speech}/metrics/"