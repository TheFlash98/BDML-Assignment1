import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/scratch/sk12184/output/checkpoint-326"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

evaluation_text = """
Climate change is caused by an increase in greenhouse gases such as CO2.
"""

tokens = tokenizer(evaluation_text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**tokens, labels=tokens["input_ids"])
    loss = outputs.loss.item()

perplexity = math.exp(loss)
print(f"Perplexity: {perplexity}")
