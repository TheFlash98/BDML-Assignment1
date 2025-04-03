import os
import glob
from torch.utils.data import Dataset
import torch
import json
from torch.nn.utils.rnn import pad_sequence

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
        print(f"input_ids: {input_ids}, labels: {labels}, attention_mask: {attention_mask}")
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
    
    def collate_fn(self, batch, pad_token_id=0, label_pad_token_id=-100):
        input_ids = [sample["input_ids"] for sample in batch]
        labels = [sample["labels"] for sample in batch]
        attention_masks = [sample["attention_mask"] for sample in batch]

        # Pad the sequences to the maximum length in the batch.
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=label_pad_token_id)
        attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_masks_padded,
        }
