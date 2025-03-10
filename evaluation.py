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
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    input_ids = tokens["input_ids"].squeeze(0)  # Remove batch dimension
    attention_mask = tokens["attention_mask"].squeeze(0)  # Remove batch dimension
    
    ans = 0
    num_chunks = 0
    max_length = 2048
    stride = 512
    
    # Create chunks with overlap
    for i in range(0, len(input_ids), max_length - stride):
        chunk_input_ids = input_ids[i: i + max_length]
        chunk_attention_mask = attention_mask[i: i + max_length]
        
        # Skip very short chunks
        if len(chunk_input_ids) < 10:
            continue
        
        # Pad the chunk if it's smaller than max_length
        if len(chunk_input_ids) < max_length:
            padding_length = max_length - len(chunk_input_ids)
            chunk_input_ids = torch.cat([chunk_input_ids, torch.zeros(padding_length, dtype=torch.long).to(device)])
            chunk_attention_mask = torch.cat([chunk_attention_mask, torch.zeros(padding_length, dtype=torch.long).to(device)])
        
        # Add batch dimension
        chunk_input_ids = chunk_input_ids.unsqueeze(0)
        chunk_attention_mask = chunk_attention_mask.unsqueeze(0)
        
        chunk = {
            "input_ids": chunk_input_ids,
            "attention_mask": chunk_attention_mask,
            "labels": chunk_input_ids
        }
        
        with torch.no_grad():
            outputs = model(**chunk)
            loss = outputs.loss.item()
            ans += math.exp(loss)  # Perplexity = e^(loss)
            num_chunks += 1
    
    return ans / num_chunks if num_chunks > 0 else float("inf")

# Compute perplexity over all test files
total_perplexity = 0
num_samples = 0

for file_path in tqdm(test_files):
    with open(file_path, "r") as f:
        text = f.read()
        if text:
            ppl = compute_perplexity(text)
            print('file path:', file_path)
            print('ppl:', ppl)
            total_perplexity += ppl
            num_samples += 1

# Average Perplexity
avg_perplexity = total_perplexity / num_samples if num_samples > 0 else float("inf")
print(f"Average Perplexity on Test Set: {avg_perplexity:.4f}")
