import torch
import json
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# Path to your fine-tuned model
model_path = "/scratch/sk12184/output/checkpoint-326"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# Path to test JSONL files
# Directory containing test JSONL files
test_dir = "/scratch/sk12184/climate_text_dataset_tokenized/eval/"

# Get all .jsonl files in the directory
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".jsonl")]

# Function to compute perplexity
def compute_perplexity(sample):
    # inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
    labels = torch.tensor(sample["labels"], dtype=torch.long)
    # Use provided attention_mask or generate one if not available
    attention_mask = sample.get("attention_mask", [1] * len(sample["input_ids"]))
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    tokens =  {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }
    with torch.no_grad():
        outputs = model(**tokens)
        loss = outputs.loss.item()
    return math.exp(loss)  # Perplexity = e^(loss)

# Compute perplexity over all test files
total_perplexity = 0
num_samples = 0

for file_path in tqdm(test_files):
    with open(file_path, "r") as f:
        text = f.read()
        json_obj = json.loads(text)  # Each line contains one JSON
        # text = json_obj.get("text", "")  # Assuming the text is stored under the "text" key
        if json_obj:
            total_perplexity += compute_perplexity(json_obj)
            num_samples += 1

# Average Perplexity
avg_perplexity = total_perplexity / num_samples if num_samples > 0 else float("inf")
print(f"Average Perplexity on Test Set: {avg_perplexity:.4f}")
