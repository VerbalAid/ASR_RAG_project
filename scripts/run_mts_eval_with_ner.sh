#!/usr/bin/env bash
# Run MTS evaluation with NER (spaCy). Requires Python 3.11 or 3.12; spaCy does not support 3.14 yet.
# Usage: from project root, run:  bash scripts/run_mts_eval_with_ner.sh

set -e
cd "$(dirname "$0")/.."

# Prefer project venv if it's Python 3.11 or 3.12
if [[ -x .venv/bin/python ]]; then
    ver=$(.venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
    if [[ "$ver" == 3.1[12] ]]; then
        echo "Using .venv/bin/python ($ver) for mts_eval (NER requires spaCy on Python 3.11/3.12)."
        .venv/bin/python -m analysis.mts_eval
        exit 0
    fi
fi

for py in python3.12 python3.11 python; do
    if command -v "$py" >/dev/null 2>&1; then
        ver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        if [[ "$ver" == 3.1[12] ]]; then
            echo "Using $py ($ver) for mts_eval (NER requires spaCy on Python 3.11/3.12)."
            "$py" -m analysis.mts_eval
            exit 0
        fi
    fi
done

echo "No Python 3.11 or 3.12 found. NER (spaCy) requires one of these."
echo "Install: e.g.  sudo dnf install python3.11  (Fedora)  or  pyenv install 3.11  then create a venv."
echo "Then:  pip install spacy && python -m spacy download en_core_web_sm"
echo "Then re-run:  bash scripts/run_mts_eval_with_ner.sh"
exit 1
