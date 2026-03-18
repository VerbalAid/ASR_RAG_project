import whisper
import json
import soundfile as sf
import numpy as np
import io
import os
from datasets import load_dataset, Audio

ds = load_dataset("hf-audio/asr-leaderboard-longform", "tedlium", split="test")
ds = ds.cast_column("audio", Audio(decode=False))

model = whisper.load_model("tiny")

AUDIO_OUT_DIR = "datasets/ted_audio_samples/audio_samples"
os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
os.makedirs("results/ted", exist_ok=True)

results = []
for i in range(len(ds)):
    print(f"Processing sample {i+1}/11...")
    sample = ds[i]

    audio_bytes = sample['audio']['bytes']
    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # Save audio file (TED samples under datasets/ted_audio_samples)
    audio_filename = f"{AUDIO_OUT_DIR}/sample_{i}_{sample['speaker_id']}.wav"
    sf.write(audio_filename, audio_array, sample_rate)

    # Convert to mono float32 for Whisper
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32)

    result = model.transcribe(audio_array)

    results.append({
        "sample_id": i,
        "speaker_id": sample['speaker_id'],
        "audio_file": audio_filename,
        "ground_truth": sample['text'],
        "c1_whisper_tiny": result['text']
    })
    print(f"Done sample {i+1}")

with open("results/ted/c1_whisper_raw.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved to results/ted/c1_whisper_raw.json")
