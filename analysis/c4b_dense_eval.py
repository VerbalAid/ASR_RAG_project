import json
import re
from pathlib import Path
from jiwer import wer
from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

Path("results/ted/metrics").mkdir(parents=True, exist_ok=True)
with open('results/ted/c4_den_mat_outputs.json', 'r') as f:
    results = json.load(f)

def normalise(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

scorer    = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoothing = SmoothingFunction().method1
metrics_results = []

for sample in results:
    gt = normalise(sample['ground_truth'])
    c4 = normalise(sample['c4b_rag_ted'] or '')
    if not c4:
        print(f"SKIP {sample['speaker_id']} — empty")
        continue

    wer_score  = wer(gt, c4)
    bleu_score = sentence_bleu([gt.split()], c4.split(), smoothing_function=smoothing)
    rouge_l    = scorer.score(gt, c4)['rougeL'].fmeasure
    P, R, F1   = score([c4], [gt], lang='en', verbose=False)
    bert       = F1.mean().item()

    metrics_results.append({
        'sample_id':  sample['sample_id'],
        'speaker_id': sample['speaker_id'],
        'wer':        round(wer_score,  4),
        'bleu':       round(bleu_score, 4),
        'rouge_l':    round(rouge_l,    4),
        'bert_score': round(bert,       4)
    })
    print(f"{sample['speaker_id']:<30} WER: {wer_score:.3f} | BLEU: {bleu_score:.3f} | ROUGE-L: {rouge_l:.3f} | BERT: {bert:.3f}")

avg = lambda k: sum(r[k] for r in metrics_results) / len(metrics_results)
print(f"\n--- C4b TED RAG AVERAGES ---")
print(f"WER:       {avg('wer'):.4f}  (C1: 0.075 | C2: 0.387 | C3: 0.245)")
print(f"BLEU:      {avg('bleu'):.4f}  (C1: 0.867 | C2: 0.482 | C3: 0.674)")
print(f"ROUGE-L:   {avg('rouge_l'):.4f}  (C1: 0.943 | C2: 0.725 | C3: 0.824)")
print(f"BERTScore: {avg('bert_score'):.4f}  (C1: 0.975 | C2: 0.925 | C3: 0.947)")

with open('results/ted/metrics/c4_den_mat_metrics.json', 'w') as f:
    json.dump(metrics_results, f, indent=2)
print("\nSaved to results/ted/metrics/c4_den_mat_metrics.json")