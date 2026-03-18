import ollama
import json
import re

with open('results/ted/c1_whisper_raw.json', 'r') as f:
    results = json.load(f)

def chunk_text(text, max_words=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(' '.join(words[i:i+max_words]))
    return chunks

def correct_transcript(text):
    chunks = chunk_text(text, max_words=400)
    corrected_chunks = []
    
    for chunk in chunks:
        response = ollama.chat(
            model='mistral:latest',
            messages=[
                {
                    'role': 'user',
                    'content': (
                        f"Fix only clear ASR errors in this transcript excerpt. "
                        f"Output ONLY the corrected text. No commentary, no notes, no preamble.\n\n"
                        f"{chunk}"
                    )
                }
            ]
        )
        raw = response['message']['content'].strip()
        # Strip any meta-commentary
        raw = re.sub(r'^(here is|corrected transcript|note:).{0,100}\n', '', raw, flags=re.IGNORECASE)
        corrected_chunks.append(raw.strip())
    
    return ' '.join(corrected_chunks)

for i, result in enumerate(results):
    print(f"Processing {i+1}/{len(results)}: {result['speaker_id']}...")
    corrected = correct_transcript(result['c1_whisper_tiny'])
    
    input_len = len(result['c1_whisper_tiny'].split())
    output_len = len(corrected.split())
    print(f"  {input_len}w → {output_len}w (ratio: {output_len/input_len:.2f})")
    
    result['c2_llm_only'] = corrected
    result['c2_flag'] = 'ok'

with open('results/ted/c2_outputs_mistral.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nDone. Saved to results/ted/c2_outputs_mistral.json")