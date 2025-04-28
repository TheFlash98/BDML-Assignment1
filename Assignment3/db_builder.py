# RAG System - Database Building Code
# This script demonstrates how to build a vector database for a RAG system

import os
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Union
import faiss
import time
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Class for loading and processing documents"""
    
    def __init__(self, data_dir: str):
        """
        Initialize the DocumentProcessor
        
        Args:
            data_dir: Directory containing the documents
        """
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # tokens per chunk
            chunk_overlap=50,  # token overlap between chunks
            length_function=len,
            is_separator_regex=False,
        )
        
    def load_documents(self) -> List[Dict]:
        """
        Load documents from directory
        
        Returns:
            List of document dictionaries with text and metadata
        """
        logger.info(f"Loading documents from {self.data_dir}")
        documents = []
        
        # Handle text files
        text_loader = DirectoryLoader(self.data_dir, glob="**/*.txt", loader_cls=TextLoader)
        text_docs = text_loader.load()
        
        all_docs = text_docs
        logger.info(f"Loaded {len(all_docs)} documents")
        
        for doc in all_docs:
            documents.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })
            
        return documents
    
    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split documents into chunks
        
        Args:
            documents: List of document dictionaries
        
        Returns:
            List of document chunk dictionaries
        """
        logger.info("Splitting documents into chunks")
        chunks = []
        
        for doc in documents:
            doc_chunks = self.text_splitter.split_text(doc["text"])
            
            for i, chunk_text in enumerate(doc_chunks):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_id": i
                    }
                })
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

class EmbeddingModel:
    """Class for generating embeddings using a transformer model"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name of the Hugging Face model to use
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model loaded and moved to {self.device}")
        
    def get_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        self.model.eval()
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            # Mean pooling - use attention mask to calculate mean
            attention_mask = encoded_input['attention_mask']
            token_embeddings = model_output[0]  # First element of model_output contains token embeddings
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
            # Normalize embeddings
            batch_embeddings = normalize(batch_embeddings, norm='l2', axis=1)
            
            embeddings.append(batch_embeddings)
            
        # Combine all batches
        all_embeddings = np.vstack(embeddings)
        logger.info(f"Generated embeddings with shape {all_embeddings.shape}")
        
        return all_embeddings

class VectorDatabase:
    """Class for creating and managing a FAISS vector database"""
    
    def __init__(self, embedding_dim: int = 384, index_type: str = "HNSW"):
        """
        Initialize the vector database
        
        Args:
            embedding_dim: Dimension of the embeddings
            index_type: Type of index to use ('Flat', 'IVF', 'HNSW', 'PQ')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.chunk_data = []
        
    def create_index(self, embeddings: np.ndarray):
        """
        Create a FAISS index from embeddings
        
        Args:
            embeddings: Array of embeddings
        """
        logger.info(f"Creating {self.index_type} index for {len(embeddings)} vectors")
        
        if self.index_type == "Flat":
            # Exact search (slowest but most accurate)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
        elif self.index_type == "IVF":
            # Inverted file index (faster approximate search)
            nlist = min(4096, int(len(embeddings) / 10))  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
            # Train the index
            logger.info(f"Training IVF index with {nlist} clusters")
            self.index.train(embeddings)
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graph (fast approximate search)
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # M=32 (connections per node)
            
        elif self.index_type == "PQ":
            # Product Quantization (compact storage)
            m = 8  # Number of subquantizers
            bits = 8  # Bits per subquantizer (usually 8)
            
            # Create an index with a flat quantizer
            nlist = min(4096, int(len(embeddings) / 10))  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            
            # Create the IVFPQ index
            self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, bits)
            
            # Train the index
            logger.info(f"Training IVFPQ index with {nlist} clusters, {m} subquantizers")
            self.index.train(embeddings)
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Add vectors to index
        logger.info("Adding vectors to index")
        self.index.add(embeddings)
        logger.info(f"Created index with {self.index.ntotal} vectors")
        
    def save(self, index_path: str, metadata_path: str):
        """
        Save the index and metadata
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the chunk metadata
        """
        logger.info(f"Saving index to {index_path}")
        faiss.write_index(self.index, index_path)
        
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(self.chunk_data, f)
            
    @classmethod
    def load(cls, index_path: str, metadata_path: str):
        """
        Load the index and metadata
        
        Args:
            index_path: Path to the FAISS index
            metadata_path: Path to the chunk metadata
            
        Returns:
            VectorDatabase instance
        """
        logger.info(f"Loading index from {index_path}")
        index = faiss.read_index(index_path)
        
        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            chunk_data = json.load(f)
            
        # Create database instance
        db = cls(embedding_dim=index.d)
        db.index = index
        db.chunk_data = chunk_data
        
        return db

def build_database(
    data_dir: str,
    output_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_type: str = "HNSW"
):
    """
    Build a vector database from documents
    
    Args:
        data_dir: Directory containing the documents
        output_dir: Directory to save the database
        model_name: Name of the embedding model
        index_type: Type of FAISS index to use
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process documents
    processor = DocumentProcessor(data_dir)
    documents = processor.load_documents()
    chunks = processor.split_documents(documents)
    
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings
    embedding_model = EmbeddingModel(model_name)
    embeddings = embedding_model.get_embeddings(texts)
    
    # Create vector database
    embed_dim = embeddings.shape[1]
    db = VectorDatabase(embedding_dim=embed_dim, index_type=index_type)
    db.chunk_data = chunks
    db.create_index(embeddings)
    
    # Save database
    index_path = os.path.join(output_dir, "faiss_index.bin")
    metadata_path = os.path.join(output_dir, "chunks_metadata.json")
    db.save(index_path, metadata_path)
    
    logger.info(f"Database built successfully: {len(chunks)} chunks indexed")
    
    return {
        "index_path": index_path,
        "metadata_path": metadata_path,
        "num_chunks": len(chunks),
        "embedding_dim": embed_dim
    }

if __name__ == "__main__":
    # Example usage
    DATA_DIR = "/path/to/your/data"
    OUTPUT_DIR = "./rag_database"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Can also use "BAAI/bge-large-en" for better quality
    INDEX_TYPE = "HNSW"  # Options: "Flat", "IVF", "HNSW", "PQ"
    
    start_time = time.time()
    results = build_database(DATA_DIR, OUTPUT_DIR, MODEL_NAME, INDEX_TYPE)
    end_time = time.time()
    
    print(f"Database built in {end_time - start_time:.2f} seconds")
    print(f"Number of chunks indexed: {results['num_chunks']}")
    print(f"Embedding dimension: {results['embedding_dim']}")
    print(f"Index saved to: {results['index_path']}")
    print(f"Metadata saved to: {results['metadata_path']}")