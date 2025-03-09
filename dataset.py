import os
import glob
from torch.utils.data import Dataset
import torch
import json

class ClimateDataset(Dataset):
    def __init__(self, data_root_path="climate_text_dataset_processed",split="train"):
        self.data_root_path = data_root_path
        self.split = split
        self.text_dir = os.path.join(data_root_path, split)
        self.text_files = glob.glob(os.path.join(self.text_dir, "*.jsonl"))

    def __len__(self):
        return len(self.text_files)
    
    def __getitem__(self, idx):
        text_path = self.text_files[idx]
        with open(text_path, "r") as f:
            text = f.read()
            sample = json.loads(text)
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        labels = torch.tensor(sample["labels"], dtype=torch.long)
        # Use provided attention_mask or generate one if not available
        attention_mask = sample.get("attention_mask", [1] * len(sample["input_ids"]))
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

