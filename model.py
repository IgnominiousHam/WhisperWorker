import os
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Common OPUS-MT → English models (major languages)
models = [
    "Helsinki-NLP/opus-mt-fr-en",  # French → English
    "Helsinki-NLP/opus-mt-es-en",  # Spanish → English
    "Helsinki-NLP/opus-mt-de-en",  # German → English
    "Helsinki-NLP/opus-mt-it-en",  # Italian → English
    "Helsinki-NLP/opus-mt-pt-en",  # Portuguese → English
    "Helsinki-NLP/opus-mt-ru-en",  # Russian → English
    "Helsinki-NLP/opus-mt-zh-en",  # Chinese → English
    "Helsinki-NLP/opus-mt-ja-en",  # Japanese → English
    "Helsinki-NLP/opus-mt-ar-en",  # Arabic → English
    "Helsinki-NLP/opus-mt-hi-en",  # Hindi → English
    "Helsinki-NLP/opus-mt-pl-en",  # Polish → English
    "Helsinki-NLP/opus-mt-fi-en",  # Finnish → English
]

print(f"📦 Preparing to download {len(models)} OPUS-MT models...\n")

for model_id in models:
    short = model_id.split("/")[-1]
    try:
        print(f"📥 Downloading {short} ...")
        AutoTokenizer.from_pretrained(model_id)
        AutoModelForSeq2SeqLM.from_pretrained(model_id)
        print(f"✅ Cached {short}")
        time.sleep(1)
    except Exception as e:
        print(f"⚠️ Failed for {short}: {e}")

print("\n🎉 All selected OPUS-MT → English models cached locally!")
print(f"Cache directory: {os.environ.get('TRANSFORMERS_CACHE', '~/.cache/huggingface/')}")
