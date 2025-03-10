import torch
import json
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# Path to your fine-tuned model
model_path = "/scratch/sk12184/output/checkpoint-326"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

evaluation_text = """
Climate change is caused by an increase in greenhouse gases in the atmosphere such as CO2.
"""

tokens = tokenizer(evaluation_text, return_tensors="pt", truncation=True, max_length=2048)
print(len(tokens))
print(tokens)
with torch.no_grad():
    outputs = model(**tokens)
    loss = outputs.loss.item()

perplexity = math.exp(loss)  # Perplexity = e^(loss)
print(f"Perplexity on evaluation text: {perplexity:.4f}")