import os
import glob
import shutil
import random
from pathlib import Path

def split_text_files(data_root_path, train_ratio=0.9, eval_ratio=0.1):
    """
    Split text files into train and eval folders with the specified ratio.
    Moves original text files to their respective directories.
    
    Args:
        data_root_path (str): Path to the directory containing text files
        train_ratio (float): Ratio of files to be placed in the train folder (default: 0.9)
        eval_ratio (float): Ratio of files to be placed in the eval folder (default: 0.1)
    """
    # Create train and eval directories if they don't exist
    train_dir = os.path.join(data_root_path, "train")
    eval_dir = os.path.join(data_root_path, "eval")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Get all text files in the data root directory
    text_files = glob.glob(os.path.join(data_root_path, "*.txt"))
    
    if not text_files:
        print(f"No text files found in {data_root_path}")
        return
    
    # Shuffle the files to ensure random distribution
    random.shuffle(text_files)
    
    # Calculate the split point
    split_idx = int(len(text_files) * train_ratio)
    
    # Split the files
    train_files = text_files[:split_idx]
    eval_files = text_files[split_idx:]
    
    # Move files to their respective directories
    for file_path in train_files:
        filename = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(train_dir, filename))
    
    for file_path in eval_files:
        filename = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(eval_dir, filename))
    
    print(f"Split complete: {len(train_files)} files in train, {len(eval_files)} files in eval")
    print(f"Original text files have been moved to their respective directories")

if __name__ == "__main__":
    # You can change this path to your actual data directory
    data_path = "climate_text_dataset_processed"
    split_text_files(data_path)


