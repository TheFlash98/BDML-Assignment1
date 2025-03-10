import torch
import json
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# Path to your fine-tuned model
model_path = "/scratch/sk12184/output/checkpoint-326"

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Path to test JSONL files
# Directory containing test JSONL files
test_dir = "climate_text_dataset_processed/eval"

# Get all .jsonl files in the directory
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".txt")]

# Function to compute perplexity
def compute_perplexity(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    ans = 0
    # Create chunks with overlap
    for i in range(0, len(tokens), 2048 - 512):
        chunk = tokens[i: i + 2048]
        with torch.no_grad():
            outputs = model(**chunk, labels=chunk["input_ids"])
            loss = outputs.loss.item()
            ans += math.exp(loss)  # Perplexity = e^(loss)

# Compute perplexity over all test files
total_perplexity = 0
num_samples = 0

for file_path in tqdm(test_files):
    with open(file_path, "r") as f:
        text = f.read()
        if text:
            total_perplexity += compute_perplexity(text)
            num_samples += 1

# Average Perplexity
avg_perplexity = total_perplexity / num_samples if num_samples > 0 else float("inf")
print(f"Average Perplexity on Test Set: {avg_perplexity:.4f}")
