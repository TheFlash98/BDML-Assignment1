import os
import json
from transformers import AutoTokenizer

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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    for filename in os.listdir(input_dir)[:10]:
        if filename.endswith('.txt'):
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
    input_dir = "climate_text_dataset_processed/eval"
    
    # Output JSONL file with pre-tokenized examples
    output_file = "/scratch/sk12184/climate_text_dataset_tokenized/eval"
    
    # Provide the path or model id for your LLaMA tokenizer
    tokenizer_name = "path/to/llama-tokenizer"  # e.g., "huggingface/llama-7b" if available
    
    # Maximum tokens per input (typically 2048 for LLaMA, adjust as needed)
    max_length = 2048
    
    # Stride for overlapping chunks (helps with context preservation)
    stride = 512
    
    process_directory(input_dir, output_file, tokenizer_name, max_length, stride)
    print("Tokenization complete. Data saved to:", output_file)
