import json, re
from pathlib import Path
from jiwer import wer
from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

Path("results/ted/metrics").mkdir(parents=True, exist_ok=True)
with open('results/ted/c2_outputs_mistral.json', 'r') as f:
    results = json.load(f)

def normalise(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoothing = SmoothingFunction().method1
metrics_results = []

for sample in results:
    gt = normalise(sample['ground_truth'])
    c2 = normalise(sample['c2_llm_only'] or '')
    if not c2:
        print(f"SKIP {sample['speaker_id']} — empty output")
        continue
    wer_score = wer(gt, c2)
    bleu_score = sentence_bleu([gt.split()], c2.split(), smoothing_function=smoothing)
    rouge_l = scorer.score(gt, c2)['rougeL'].fmeasure
    P, R, F1 = score([c2], [gt], lang='en', verbose=False)
    bert = F1.mean().item()
    metrics_results.append({
        'sample_id': sample['sample_id'],
        'speaker_id': sample['speaker_id'],
        'wer': round(wer_score, 4),
        'bleu': round(bleu_score, 4),
        'rouge_l': round(rouge_l, 4),
        'bert_score': round(bert, 4)
    })
    print(f"{sample['speaker_id']:<30} WER: {wer_score:.3f} | BLEU: {bleu_score:.3f} | ROUGE-L: {rouge_l:.3f} | BERT: {bert:.3f}")

avg = lambda k: sum(r[k] for r in metrics_results) / len(metrics_results)
print(f"\n--- C2 MISTRAL AVERAGES ---")
print(f"WER:       {avg('wer'):.4f}  (C1 was ~0.075)")
print(f"BLEU:      {avg('bleu'):.4f}  (C1 was ~0.867)")
print(f"ROUGE-L:   {avg('rouge_l'):.4f}  (C1 was ~0.943)")
print(f"BERTScore: {avg('bert_score'):.4f}  (C1 was ~0.975)")

with open('results/ted/metrics/c2_metrics_mistral.json', 'w') as f:
    json.dump(metrics_results, f, indent=2)
print("\nSaved to results/ted/metrics/c2_metrics_mistral.json")