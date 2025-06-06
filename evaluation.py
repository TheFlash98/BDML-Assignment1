import torch
import json
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import argparse

# Path to your fine-tuned model
# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate model perplexity on test set")
parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
args = parser.parse_args()

model_path = args.model_path
print("Evaluating model ", model_path)

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

# Path to test JSONL files
# Directory containing test JSONL files
test_dir = "climate_text_dataset_processed/eval"

# Get all .jsonl files in the directory
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".txt")]

# Function to compute perplexity
def compute_perplexity(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**tokens, labels=tokens["input_ids"])
        loss = outputs.loss
    return math.exp(loss)

# Compute perplexity over all test files
total_perplexity = 0
num_samples = 0

for file_path in tqdm(test_files):
    with open(file_path, "r") as f:
        text = f.read()
        words = text.split()
        chunk_size = 1500
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                ppl = compute_perplexity(chunk)
            total_perplexity += ppl
            num_samples += 1

# Average Perplexity
avg_perplexity = total_perplexity / num_samples if num_samples > 0 else float("inf")
print(f"Average Perplexity on Test Set: {avg_perplexity:.4f}")
