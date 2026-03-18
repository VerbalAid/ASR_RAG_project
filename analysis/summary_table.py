import json
import re
import sys
from pathlib import Path
from jiwer import wer
from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
sys.setrecursionlimit(10000)

Path("results/ted/metrics").mkdir(parents=True, exist_ok=True)

def normalise(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

with open('results/ted/c1_whisper_raw.json', 'r') as f:
    results = json.load(f)

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoothing = SmoothingFunction().method1
metrics_results = []

for sample in results:
    gt = normalise(sample['ground_truth'])
    c1 = normalise(sample['c1_whisper_tiny'])

    wer_score = wer(gt, c1)
    bleu_score = sentence_bleu([gt.split()], c1.split(), smoothing_function=smoothing)
    rouge_l = scorer.score(gt, c1)['rougeL'].fmeasure
    P, R, F1 = score([c1], [gt], lang='en', verbose=False)
    bert = F1.mean().item()

    metrics_results.append({
        'sample_id': sample['sample_id'],
        'speaker_id': sample['speaker_id'],
        'wer': round(wer_score, 4),
        'bleu': round(bleu_score, 4),
        'rouge_l': round(rouge_l, 4),
        'bert_score': round(bert, 4)
    })
    print(f"[{sample['sample_id']}] {sample['speaker_id']} | WER: {wer_score:.3f} | BLEU: {bleu_score:.3f} | ROUGE-L: {rouge_l:.3f} | BERT: {bert:.3f}")

avg = lambda key: sum(r[key] for r in metrics_results) / len(metrics_results)
print("\n--- C1 BASELINE AVERAGES ---")
print(f"WER:       {avg('wer'):.4f}")
print(f"BLEU:      {avg('bleu'):.4f}")
print(f"ROUGE-L:   {avg('rouge_l'):.4f}")
print(f"BERTScore: {avg('bert_score'):.4f}")

with open('results/ted/metrics/c1_metrics.json', 'w') as f:
    json.dump(metrics_results, f, indent=2)
print("\nSaved to results/ted/metrics/c1_metrics.json")