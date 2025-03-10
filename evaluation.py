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
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# Path to test JSONL files
# Directory containing test JSONL files
test_dir = "/scratch/sk12184/climate_text_dataset_tokenized/eval/"

# Get all .jsonl files in the directory
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".jsonl")]

# Function to compute perplexity
def compute_perplexity(json_obj):
    # inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**json_obj, labels=json_obj["input_ids"])
        loss = outputs.loss.item()
    return math.exp(loss)  # Perplexity = e^(loss)

# Compute perplexity over all test files
total_perplexity = 0
num_samples = 0

for file_path in tqdm(test_files):
    with open(file_path, "r") as f:
        for line in f:
            json_obj = json.loads(line)  # Each line contains one JSON
            # text = json_obj.get("text", "")  # Assuming the text is stored under the "text" key
            if json_obj:
                total_perplexity += compute_perplexity(json_obj)
                num_samples += 1

# Average Perplexity
avg_perplexity = total_perplexity / num_samples if num_samples > 0 else float("inf")
print(f"Average Perplexity on Test Set: {avg_perplexity:.4f}")
