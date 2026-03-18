import json
import re
from pathlib import Path
from jiwer import wer
from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ================= LOAD RESULTS (C3 BM25 on Wikipedia) =================
Path("results/ted/metrics").mkdir(parents=True, exist_ok=True)

with open('results/ted/c3_lex_gen_outputs_bm25.json', 'r') as f:
    results = json.load(f)

# ================= EXTRACT OLLAMA CONTENT (handles double‑quoted content) =================

def extract_ollama_content(raw: str) -> str:
    """
    Extract all assistant message contents from concatenated Ollama responses.
    Looks for the pattern: message=Message(role='assistant', content="...")
    The content is captured until the closing " that is followed by a comma and a known key.
    Handles multiple chunks.
    """
    if not raw:
        return ""
    # Pattern: content="...", where the closing " is the one before a comma + known key or ')'
    # Known keys: thinking, images, tool_name, tool_calls, logprobs
    pattern = r'content="(.*?)"(?=, (?:thinking|images|tool_name|tool_calls|logprobs|\)))'
    matches = re.findall(pattern, raw, flags=re.DOTALL)
    if matches:
        # Unescape any escaped quotes (JSON unescaping already happened, but sometimes there are literal backslashes)
        # We'll just join with spaces.
        return ' '.join(matches)
    # Fallback: try single‑quoted content pattern (some responses might use single quotes)
    pattern2 = r"content='((?:\\\\.|[^'\\\\])*)'"
    matches2 = re.findall(pattern2, raw, flags=re.DOTALL)
    if matches2:
        cleaned = [m.replace("\\'", "'") for m in matches2]
        return ' '.join(cleaned)
    return ""

# ================= TEXT NORMALISATION =================

def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoothing = SmoothingFunction().method1

metrics_results = []

# ================= EVALUATION LOOP =================

for sample in results:
    # Ground truth
    gt = normalise(sample.get('ground_truth', ''))
    if not gt:
        print(f"SKIP {sample.get('speaker_id')} — missing ground truth")
        continue

    # BM25 output (raw, may contain Ollama metadata)
    c3_raw = sample.get('c3_bm25_rag') or ''
    # Extract only the assistant's actual responses
    c3_content = extract_ollama_content(c3_raw)
    if not c3_content:
        print(f"SKIP {sample.get('speaker_id')} — empty after extraction")
        continue

    c3 = normalise(c3_content)

    # Compute metrics
    wer_score = wer(gt, c3)
    bleu_score = sentence_bleu([gt.split()], c3.split(), smoothing_function=smoothing)
    rouge_l = scorer.score(gt, c3)['rougeL'].fmeasure
    P, R, F1 = score([c3], [gt], lang='en', verbose=False)
    bert = F1.mean().item()

    metrics_results.append({
        'sample_id':  sample.get('sample_id'),
        'speaker_id': sample.get('speaker_id'),
        'wer':        round(wer_score, 4),
        'bleu':       round(bleu_score, 4),
        'rouge_l':    round(rouge_l, 4),
        'bert_score': round(bert, 4)
    })

    print(f"{sample.get('speaker_id'):<30} "
          f"WER: {wer_score:.3f} | "
          f"BLEU: {bleu_score:.3f} | "
          f"ROUGE-L: {rouge_l:.3f} | "
          f"BERT: {bert:.3f}")

# ================= AGGREGATES =================

if metrics_results:
    avg = lambda k: sum(r[k] for r in metrics_results) / len(metrics_results)

    print("\n--- C3 BM25 RAG AVERAGES ---")
    print(f"WER:       {avg('wer'):.4f}")
    print(f"BLEU:      {avg('bleu'):.4f}")
    print(f"ROUGE-L:   {avg('rouge_l'):.4f}")
    print(f"BERTScore: {avg('bert_score'):.4f}")
else:
    print("No valid samples evaluated.")

# ================= SAVE METRICS =================

with open('results/ted/metrics/c3_lex_gen_metrics.json', 'w') as f:
    json.dump(metrics_results, f, indent=2)

print("\nSaved to results/ted/metrics/c3_lex_gen_metrics.json")