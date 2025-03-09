import os
import json
from transformers import AutoTokenizer
import argparse

def tokenize_and_chunk(text, tokenizer, max_length=2048, stride=512):
    """
    Tokenizes the text and splits it into overlapping chunks.
    
    Args:
        text (str): The text to tokenize.
        tokenizer: A Hugging Face tokenizer.
        max_length (int): Maximum tokens per chunk.
        stride (int): Overlap between chunks.
    
    Returns:
        List[List[int]]: List of token id chunks.
    """
    # Tokenize the text
    tokenized = tokenizer.encode(text, add_special_tokens=True)
    chunks = []
    
    # Create chunks with overlap
    for i in range(0, len(tokenized), max_length - stride):
        chunk = tokenized[i: i + max_length]
        if len(chunk) < 10:  # Skip very short chunks
            continue
        if len(chunk) < max_length:
            chunk += [tokenizer.pad_token_id] * (max_length - len(chunk))
        chunks.append(chunk)
    return chunks

def process_directory(input_dir, output_dir, tokenizer_name, max_length=2048, stride=512):
    """
    Processes all .txt files in a directory, tokenizes them,
    and writes the tokenized chunks to a JSONL file.
    
    Args:
        input_dir (str): Directory with text files.
        output_dir (str): Output JSONL directory.
        tokenizer_name (str): Pretrained tokenizer identifier or path.
        max_length (int): Maximum token length for each chunk.
        stride (int): Overlap between chunks.
    """
    # Initialize the tokenizer (make sure it matches your LLaMA model)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        padding_side="right",
        truncation=True,
        padding=True,
        max_length=max_length)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            print("Tokenizing", filename)
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # Tokenize and split into chunks
            chunks = tokenize_and_chunk(text, tokenizer, max_length, stride)
            for i in range(len(chunks)):
                data = {
                    "input_ids": chunks[i],
                    "labels": chunks[i]  # For causal LM tasks, labels mirror input_ids.
                }
                with open(os.path.join(output_dir,
                                       filename.replace('.txt', f'_{i}.jsonl')),
                                       'w',
                                       encoding='utf-8') as out_f:
                    out_f.write(json.dumps(data) + '\n')
                    
if __name__ == "__main__":
    # Directory containing your .txt files of research papers
    parser = argparse.ArgumentParser(description="Tokenize and chunk text files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .txt files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save tokenized JSONL files.")

    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    tokenizer_name = args.tokenizer_name
    max_length = args.max_length
    stride = args.stride
    
    # Provide the path or model id for your LLaMA tokenizer
    tokenizer_name = "/scratch/sk12184/llama3.2-3B-HF/"  # e.g., "huggingface/llama-7b" if available
    
    # Maximum tokens per input (typically 2048 for LLaMA, adjust as needed)
    max_length = 1730
    
    # Stride for overlapping chunks (helps with context preservation)
    stride = 512
    
    process_directory(input_dir, output_dir, tokenizer_name, max_length, stride)
    print("Tokenization complete. Data saved to:", output_dir)
