import os
import glob
from PyPDF2 import PdfReader
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    reader = PdfReader(pdf_path)
    text = ""
    # Add progress bar for pages in the PDF
    for page in tqdm(reader.pages, desc=f"Reading {os.path.basename(pdf_path)}", leave=False):
        text += page.extract_text()
    return text

def process_single_pdf(pdf_path, target_dir):
    """
    Process a single PDF file and save extracted text to the target directory.
    
    Args:
        pdf_path (str): Path to the PDF file
        target_dir (str): Path to the target directory where text file will be saved
        
    Returns:
        int: 1 if processed, 0 if skipped
    """
    # Create text file with the same name but .txt extension
    file = os.path.basename(pdf_path)
    txt_filename = os.path.splitext(file)[0] + '.txt'
    txt_path = os.path.join(target_dir, txt_filename)
    
    # Skip if the text file already exists
    if os.path.exists(txt_path):
        return 0
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Save the extracted text
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return 1

def process_pdfs(source_dir, target_dir, num_workers=None):
    """
    Process all PDFs in the source directory and save extracted text to the target directory.
    Skip files that have already been processed.
    
    Args:
        source_dir (str): Path to the source directory containing PDF files
        target_dir (str): Path to the target directory where text files will be saved
        num_workers (int, optional): Number of worker processes. Defaults to CPU count.
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all PDF files in the source directory
    pdf_files = glob.glob(os.path.join(source_dir, "*.pdf"))
    
    # Use all available CPU cores if not specified
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Create a partial function with the target directory
    process_func = partial(process_single_pdf, target_dir=target_dir)
    
    # Process PDFs in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, pdf_files),
            total=len(pdf_files),
            desc="Converting PDFs to text"
        ))
    
    # Count processed and skipped files
    processed_files = sum(results)
    skipped_files = len(pdf_files) - processed_files
    
    return processed_files, skipped_files

if __name__ == "__main__":
    source_directory = "climate_text_dataset"
    target_directory = "climate_text_dataset_processed"
    
    # You can specify the number of worker processes
    # For example: process_pdfs(source_directory, target_directory, num_workers=4)
    processed, skipped = process_pdfs(source_directory, target_directory)
    print(f"Text extraction complete. Processed {processed} files, skipped {skipped} already processed files.")
    print(f"Processed text saved to {target_directory}")
