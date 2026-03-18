# ASR–RAG Project

Automatic Speech Recognition (ASR) post-correction with retrieval-augmented generation (RAG). This repository compares **baseline ASR**, **LLM-only correction**, and **RAG-based correction** (lexical BM25 and dense retrieval) on three setups:

- **TED-LIUM** — long-form speech
- **MTS-Dialog** — simulated noisy clinical dialogue  
- **PriMock57 speech** — real clinical consultation audio

---

## Table of contents

1. [Project design](#1-project-design)
2. [Repository structure](#2-repository-structure)
3. [Metrics](#3-metrics)
4. [Experiments and conditions](#4-experiments-and-conditions)
5. [Setup](#5-setup)
6. [How to run](#6-how-to-run)
7. [Results summary](#7-results-summary)
8. [Visualisations](#8-visualisations)
9. [References](#9-references)

---

## 1. Project design

### Goal

Evaluate how **RAG** improves ASR post-correction. The project compares:

- **Retrieval modality:** lexical (BM25) vs dense (sentence-transformers + FAISS)
- **Corpus type:** generic (out-of-domain), domain-relevant, and domain-matched (leave-one-out)

All correction uses **Mistral 7B via Ollama** (`mistral:latest`).

### 2×3 condition grid

| Modality | Generic | Domain-relevant | Domain-matched |
|----------|---------|-----------------|----------------|
| **BM25** | C3-Lex-Gen | C3-Lex-Rel | C3-Lex-Mat |
| **Dense** | C4-Den-Gen | C4-Den-Rel | C4-Den-Mat |

- **Generic:** Wikipedia (TED), AG News (MTS)
- **Domain-relevant:** In-domain but not matched to the test sample
- **Domain-matched:** Leave-one-out (TED: exclude test speaker; MTS: exclude test dialogue)

**TED** has no C4-Den-Rel. All other cells are filled.

### Baselines

- **C1:** Raw ASR (no correction)
- **C2a:** LLaMA-only (TED; legacy)
- **C2b:** Mistral-only (LLM-only baseline)

---

## 2. Repository structure

```
ASR_RAG_project/
├── README.md
├── run_all.sh              # Full pipeline: TED + MTS + PriMock57 (prepare → experiments → eval → figures)
├── requirements.txt
│
├── experiments/
│   ├── test/               # TED (Whisper, C2, C3, C4)
│   ├── mts/                # MTS-Dialog (noise + correction)
│   └── primock57_speech/   # PriMock57 (Whisper + RAG on real audio)
│
├── analysis/               # Evaluation (WER, BLEU, ROUGE-L, BERTScore, NER)
├── visualisations/         # Figures → images/
├── scripts/                # run_visualisations.sh, run_full_mts_analysis_with_scispacy.sh, run_mts_eval_with_ner.sh
│
├── results/
│   ├── ted/                # TED outputs + metrics
│   ├── mts/                # MTS outputs + metrics + per_dialogue
│   ├── primock57/          # Prepared passages
│   ├── primock57_speech/   # PriMock57 speech outputs + metrics
│   └── wiki/               # Wikipedia cache
│
├── images/                 # Generated figures
└── datasets/               # TED audio, MTS loaders, PriMock57 raw data
```

---

## 3. Metrics

All metrics use normalised text (lowercase, strip punctuation, collapse whitespace).

| Metric | Range | Description |
|--------|------|-------------|
| **WER** | [0, ∞) | Word Error Rate (jiwer). **Lower** is better. |
| **BLEU** | [0, 1] | N-gram precision (nltk). **Higher** is better. |
| **ROUGE-L** | [0, 1] | Longest common subsequence F1. **Higher** is better. |
| **BERTScore** | [0, 1] | BERT embedding similarity. **Higher** is better. |
| **NER F1** (MTS) | [0, 1] | spaCy `en_core_web_sm`. **Higher** is better. |
| **CHEMICAL_F1 / DISEASE_F1** (MTS) | [0, 1] | scispaCy `en_ner_bc5cdr_md`. Optional. **Higher** is better. |

---

## 4. Experiments and conditions

### TED (TED-LIUM)

| ID | Condition | Corpus |
|----|-----------|--------|
| C1 | Whisper baseline | — |
| C2a | LLaMA only | — |
| C2b | Mistral only | — |
| C3-Lex-Gen | BM25 RAG | Wikipedia |
| C3-Lex-Rel | BM25 RAG | Full TED ground truth |
| C3-Lex-Mat | BM25 RAG | Leave-one-speaker-out TED |
| C4-Den-Gen | Dense RAG | Wikipedia |
| C4-Den-Mat | Dense RAG | Leave-one-speaker-out TED |

**Note:** C3-Lex-Mat can skip EricMead_2009P_EricMead if output is too short (word ratio < 0.5); heatmap shows "—".

### MTS-Dialog (simulated noise)

| ID | Condition | Corpus |
|----|-----------|--------|
| C1 | Noisy baseline | — |
| C2a | LLaMA only | — |
| C2b | Mistral only | — |
| C3-Lex-Gen | BM25 RAG | AG News |
| C3-Lex-Rel | BM25 RAG | PriMock57 passages |
| C3-Lex-Mat | BM25 RAG | Leave-one-dialogue-out clinical notes |
| C4-Den-Gen | Dense RAG | AG News |
| C4-Den-Rel | Dense RAG | PriMock57 |
| C4-Den-Mat | Dense RAG | MTS clinical notes (`section_text`) |

C4-Den-Mat WER can be inflated by verbosity; BLEU/ROUGE-L/BERTScore are more reliable.

### PriMock57 speech (real audio, n=10)

| ID | Condition | Corpus |
|----|-----------|--------|
| C1 | Whisper (tiny) | — |
| C2a | Llama only | — |
| C2b | Mistral only | — |
| C3-Lex-Gen | BM25 RAG | Wikipedia |
| C3-Lex-Rel | BM25 RAG | MTS-Dialog clean transcripts |
| C3-Lex-Mat | BM25 RAG | PriMock57 notes (leave-one-out) |
| C4-Den-Gen | Dense RAG | Wikipedia |
| C4-Den-Mat | Dense RAG | MTS-Dialog clean transcripts |

Ground truth: Praat TextGrid transcriptions. See `results/primock57_speech/PRIMOCK57_SPEECH_REPORT.md`.

---

## 5. Setup

- **Python:** 3.10–3.12 (spaCy/scispaCy do not support 3.14).
- **Ollama:** `mistral`, `llama3:8b` (for C2a).
- **Create a venv and install deps:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

- **Optional (MTS CHEMICAL/DISEASE F1):** `pip install scispacy` and  
  `pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz`  
  then `python -m spacy download en_ner_bc5cdr_md`.
- **Fedora (if scispaCy build fails):** `sudo dnf install python3.12 python3.12-devel` then recreate the venv.

Experiment runs (e.g. `experiments.mts.run`) do not require spaCy. Per-dialogue evals and Wilcoxon need `mts_eval`, which needs Python 3.10–3.12 and spaCy.

---

## 6. How to run

### Full pipeline (TED + MTS + PriMock57)

From project root:

```bash
bash run_all.sh
```

Runs: Ollama warmup → MTS C4-Den-Gen → BM25 conditions (TED + MTS) → TED evals → MTS eval → sanity check → Wilcoxon → figures → PriMock57 prepare → PriMock57 C1 (Whisper, 10 consultations) → PriMock57 RAG → PriMock57 eval → PriMock57 figure. Metrics in `results/{ted,mts,primock57_speech}/metrics/`.

### MTS with scispaCy (CHEMICAL/DISEASE F1)

```bash
bash scripts/run_full_mts_analysis_with_scispacy.sh
```

### Visualisations only

```bash
bash scripts/run_visualisations.sh
```

### Individual commands

| Pipeline | Command |
|----------|---------|
| PriMock57 prepare | `python -m analysis.prepare_primock57` |
| PriMock57 C1 | `python -m experiments.primock57_speech.c1_whisper` |
| PriMock57 RAG | `python -m experiments.primock57_speech.run_rag` |
| PriMock57 eval | `python -m analysis.primock57_speech_eval` |
| MTS eval | `python -m analysis.mts_eval` |
| MTS Wilcoxon | `python -m analysis.wilcoxon_mts` |

---

## 7. Results summary

### TED (n=10)

| Condition | WER ↓ | BLEU ↑ | ROUGE-L ↑ | BERTScore ↑ |
|-----------|--------|--------|------------|-------------|
| C1 (Whisper) | 0.079 | 0.866 | 0.943 | 0.974 |
| C2b (Mistral) | 0.305 | 0.592 | 0.790 | 0.935 |
| C4-Den-Mat | **0.140** | **0.807** | **0.909** | **0.959** |

C4-Den-Mat significantly beats C2b (Wilcoxon p < 0.01). C3-Lex-Gen has worst WER.

### MTS (n=100)

| Condition | WER ↓ | BLEU ↑ | CHEMICAL_F1 ↑ | DISEASE_F1 ↑ |
|-----------|--------|--------|---------------|--------------|
| C1 (Noisy) | **0.157** | **0.694** | **0.918** | **0.922** |
| C2b (Mistral) | 0.241 | 0.658 | 0.933 | 0.935 |
| C4-Den-Mat | 10.25* | 0.513 | 0.890 | 0.837 |

*C4-Den-Mat WER inflated by verbosity. ScispaCy: C4-Den-Mat significantly worse than C2b on DISEASE_F1 (p = 0.0013).

### PriMock57 speech (n=10)

| Condition | Mean WER |
|-----------|----------|
| C2b (Mistral) | **0.869** (best) |
| C4-Den-Mat | 0.991 |

LLM-only (Mistral) beats RAG on real clinical audio in this setup.

---

## 8. Visualisations

| Script | Output | Description |
|--------|--------|-------------|
| `ted_summary.py` | `ted_cond_metrics.png` | TED condition-level WER, BLEU, ROUGE-L, BERTScore |
| `ted_wer_heat.py` | `ted_wer_heat.png` | Per-speaker WER heatmap |
| `improvement_chart.py` | `improvement_chart.png` | WER improvement vs LLM-only (TED, 8 conditions) |
| `retrieval_luck.py` | `retrieval_luck.png` | Per-speaker WER variability (C3-Lex-Gen vs C4-Den-Mat) |
| `mts_summary.py` | `mts_cond_metrics.png` | MTS condition-level metrics (incl. scispaCy CHEMICAL/DISEASE) |
| `ner_bert_divergence.py` | `ner_bert_divergence.png` | BERTScore vs NER F1 on MTS (semantic vs entity fidelity) |
| `primock57_speech_summary.py` | `primock57_speech_cond_metrics.png` | PriMock57 WER, BLEU, ROUGE-L, BERTScore |

---

## 9. References

- **WER:** jiwer
- **BLEU:** Papineni et al., ACL 2002
- **ROUGE:** Lin, 2004
- **BERTScore:** Zhang et al., ICLR 2020
- **scispaCy:** Neumann et al., AllenNLP
- **TED-LIUM:** Rousseau et al., LREC 2012
- **MTS-Dialog:** Hugging Face `har1/MTS_Dialogue-Clinical_Note`
- **PriMock57:** Babylon Health (GitHub)
