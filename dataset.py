import os
import glob
from torch.utils.data import Dataset

class ClimateDataset(Dataset):
    def __init__(self, data_root_path="climate_text_dataset_processed",split="train"):
        self.data_root_path = data_root_path
        self.split = split
        self.text_dir = os.path.join(data_root_path, split)
        self.text_files = glob.glob(os.path.join(self.text_dir, "*.txt"))

    def __len__(self):
        return len(self.text_files)
    
    def __getitem__(self, idx):
        text_path = self.text_files[idx]
        with open(text_path, "r") as f:
            text = f.read()
        return text
    
    def collate_fn(self, batch):
        return self.tokenizer.apply_chat_template(batch, return_tensors="pt")
