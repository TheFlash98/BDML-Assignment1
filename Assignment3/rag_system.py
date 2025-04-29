# RAG System - Retrieval and Generation Components

import os
import torch
import numpy as np
import faiss
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import logging
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.preprocessing import normalize
import pandas as pd
from tqdm.notebook import tqdm  # For better display in notebooks/Colab

# Configure logging for better display in Colab
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Class for generating embeddings"""
    
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
        logger.info(f"Embedding model loaded and moved to {self.device}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        return self.get_embeddings([text])[0]
    
    def get_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        self.model.eval()
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
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
            token_embeddings = model_output[0]  # First element contains token embeddings
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
            # Normalize embeddings
            batch_embeddings = normalize(batch_embeddings, norm='l2', axis=1)
            
            embeddings.append(batch_embeddings)
            
        # Combine all batches
        all_embeddings = np.vstack(embeddings)
        
        return all_embeddings

class DocumentRetriever:
    """Class for retrieving relevant documents from the vector database"""
    
    def __init__(self, index_path: str, metadata_path: str, embedding_model: EmbeddingModel):
        """
        Initialize the document retriever
        
        Args:
            index_path: Path to the FAISS index
            metadata_path: Path to the chunk metadata
            embedding_model: Model for generating query embeddings
        """
        logger.info(f"Loading index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        logger.info(f"Loading metadata from {metadata_path}")
        self.metadata = self._load_metadata(metadata_path)
        
        self.embedding_model = embedding_model
    
    def _load_metadata(self, metadata_path: str) -> List[Dict]:
        """
        Load chunk metadata from file
        
        Args:
            metadata_path: Path to metadata file (JSON or JSONL)
            
        Returns:
            List of chunk metadata dictionaries
        """
        metadata = []
        
        # Check if the file is JSONL (line-by-line JSON)
        if metadata_path.endswith('.jsonl'):
            with open(metadata_path, 'r') as f:
                for line in f:
                    if line.strip():
                        metadata.append(json.loads(line))
        else:
            # Assume regular JSON
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return metadata
    
    def search(self, query: str, k: int = 5, rerank: bool = False) -> Tuple[List[Dict], List[float]]:
        """
        Search for most relevant chunks for a query
        
        Args:
            query: Query string
            k: Number of chunks to retrieve
            rerank: Whether to apply reranking
            
        Returns:
            List of chunk dictionaries and their similarity scores
        """
        logger.info(f"Searching for: {query}")
        
        # Get query embedding
        query_embedding = self.embedding_model.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search index
        scores, indices = self.index.search(query_embedding, k=k)
        
        # Get retrieved chunks
        chunks = []
        similarities = []
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # FAISS returns -1 if fewer than k results found
                # Calculate similarity score (cosine similarity)
                similarity = 1.0 - score / 2.0
                similarities.append(similarity)
                chunks.append(self.metadata[idx])
                logger.info(f"Retrieved chunk {i+1}: Similarity={similarity:.4f}, Source={self.metadata[idx]['metadata'].get('source', 'unknown')}")
        
        # Apply reranking if requested
        if rerank and chunks:
            chunks, similarities = self._rerank(query, chunks, similarities)
        
        return chunks, similarities
    
    def _rerank(self, query: str, chunks: List[Dict], similarities: List[float]) -> Tuple[List[Dict], List[float]]:
        """
        Rerank retrieved chunks for better relevance
        
        Args:
            query: Query string
            chunks: List of retrieved chunks
            similarities: List of similarity scores
            
        Returns:
            Reranked chunks and similarity scores
        """
        logger.info("Applying reranking")
        
        # A simple implementation of reranking using term overlap
        # For more advanced reranking, you could use a cross-encoder model
        reranked_items = []
        
        query_terms = set(query.lower().split())
        
        for chunk, similarity in zip(chunks, similarities):
            text = chunk["text"].lower()
            
            # Count term overlap
            chunk_terms = set(text.split())
            overlap = len(query_terms.intersection(chunk_terms))
            
            # Adjust similarity score with term overlap
            adjusted_score = similarity * (1 + 0.1 * overlap)
            
            reranked_items.append((chunk, adjusted_score))
        
        # Sort by adjusted score in descending order
        reranked_items.sort(key=lambda x: x[1], reverse=True)
        
        # Unzip the sorted items
        reranked_chunks, reranked_similarities = zip(*reranked_items) if reranked_items else ([], [])
        
        return list(reranked_chunks), list(reranked_similarities)

class LLaMAGenerator:
    """Class for generating responses using LLaMA"""
    
    def __init__(self, model_path: str = "/scratch/BDML25SP/llama-3b"):
        """
        Initialize the LLaMA model
        
        Args:
            model_path: Path to the pretrained or fine-tuned LLaMA model
        """
        logger.info(f"Loading LLaMA model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Configure model loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # For better performance on limited GPU memory
        load_config = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        
        if device == "cuda":
            # Check available GPU memory and configure accordingly
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                **load_config,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
            )
        
        logger.info(f"LLaMA model loaded and moved to {device}")
    
    def generate(self, 
                prompt: str, 
                max_tokens: int = 256, 
                temperature: float = 0.7,
                top_p: float = 0.9,
                repetition_penalty: float = 1.1) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling probability
            repetition_penalty: Penalty for repetition
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate text
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        end_time = time.time()
        logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
        
        return generated_text

class RAGSystem:
    """Complete RAG System combining retrieval and generation"""
    
    def __init__(self, 
                index_path: str, 
                metadata_path: str, 
                embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                llama_model_path: str = "/scratch/BDML25SP/llama-3b"):
        """
        Initialize the RAG system
        
        Args:
            index_path: Path to the FAISS index
            metadata_path: Path to the chunk metadata
            embedding_model_name: Name of the embedding model
            llama_model_path: Path to the LLaMA model
        """
        # Initialize embedding model
        self.embedding_model = EmbeddingModel(embedding_model_name)
        
        # Initialize document retriever
        self.retriever = DocumentRetriever(index_path, metadata_path, self.embedding_model)
        
        # Initialize generator
        self.generator = LLaMAGenerator(llama_model_path)
    
    def answer_question(self, 
                        question: str, 
                        k: int = 5, 
                        rerank: bool = True,
                        show_retrieved: bool = False) -> Dict:
        """
        Answer a question using the RAG system
        
        Args:
            question: Question to answer
            k: Number of chunks to retrieve
            rerank: Whether to apply reranking
            show_retrieved: Whether to include retrieved chunks in the response
            
        Returns:
            Dictionary with answer, retrieved chunks, and performance metrics
        """
        logger.info(f"Processing question: {question}")
        
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieval_start = time.time()
        chunks, similarities = self.retriever.search(question, k=k, rerank=rerank)
        retrieval_time = time.time() - retrieval_start
        
        # Format context for LLM
        context_texts = [f"[Document {i+1}]: {chunk['text']}" for i, chunk in enumerate(chunks)]
        context = "\n\n".join(context_texts)
        
        # Create prompt for generation
        prompt = self._create_prompt(question, context)
        
        # Generate answer
        generation_start = time.time()
        answer = self.generator.generate(prompt)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        result = {
            "question": question,
            "answer": answer,
            "performance": {
                "total_time": total_time,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
            }
        }
        
        if show_retrieved:
            result["retrieved_chunks"] = [
                {
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "similarity": similarity
                }
                for chunk, similarity in zip(chunks, similarities)
            ]
        
        logger.info(f"Question answered in {total_time:.2f} seconds (retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")
        
        return result
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create a prompt for the LLM
        
        Args:
            question: Question to answer
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        return f"""You are a helpful assistant. Answer the question based on the provided context.
If the context doesn't contain relevant information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

    def benchmark(self, questions: List[str], k_values: List[int] = [3, 5, 7]) -> pd.DataFrame:
        """
        Benchmark the RAG system with different retrieval settings
        
        Args:
            questions: List of questions to benchmark
            k_values: List of k values to test
            
        Returns:
            DataFrame with benchmark results
        """
        results = []
        
        for question in tqdm(questions, desc="Benchmarking questions"):
            for k in k_values:
                for rerank in [False, True]:
                    # Run RAG with current settings
                    result = self.answer_question(question, k=k, rerank=rerank)
                    
                    # Record result
                    results.append({
                        "question": question,
                        "k": k,
                        "rerank": rerank,
                        "answer": result["answer"],
                        "total_time": result["performance"]["total_time"],
                        "retrieval_time": result["performance"]["retrieval_time"],
                        "generation_time": result["performance"]["generation_time"],
                    })
        
        return pd.DataFrame(results)

def compare_with_finetuned(rag_system: RAGSystem, finetuned_model_path: str, questions: List[str]) -> pd.DataFrame:
    """
    Compare RAG system with fine-tuned LLaMA
    
    Args:
        rag_system: Initialized RAG system
        finetuned_model_path: Path to fine-tuned LLaMA model
        questions: List of questions to compare
        
    Returns:
        DataFrame with comparison results
    """
    # Initialize fine-tuned model
    finetuned_generator = LLaMAGenerator(finetuned_model_path)
    
    results = []
    
    for question in tqdm(questions, desc="Comparing models"):
        # RAG answer
        rag_start = time.time()
        rag_result = rag_system.answer_question(question)
        rag_time = time.time() - rag_start
        
        # Fine-tuned answer
        ft_start = time.time()
        ft_prompt = f"Question: {question}\n\nAnswer:"
        ft_answer = finetuned_generator.generate(ft_prompt)
        ft_time = time.time() - ft_start
        
        results.append({
            "question": question,
            "rag_answer": rag_result["answer"],
            "ft_answer": ft_answer,
            "rag_time": rag_time,
            "ft_time": ft_time,
            "time_difference": rag_time - ft_time,
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage
    INDEX_PATH = "./rag_database/faiss_index.bin"
    METADATA_PATH = "./rag_database/chunks_metadata.jsonl"  # or .json
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLAMA_MODEL_PATH = "/scratch/BDML25SP/llama-3b"
    FINETUNED_MODEL_PATH = "/path/to/your/finetuned/llama"
    
    # Initialize RAG system
    rag = RAGSystem(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        embedding_model_name=EMBEDDING_MODEL,
        llama_model_path=LLAMA_MODEL_PATH
    )
    
    # Example question
    question = "What is the architecture of a RAG system?"
    result = rag.answer_question(question, show_retrieved=True)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Total time: {result['performance']['total_time']:.2f} seconds")
    
    # Benchmark different retrieval settings
    benchmark_questions = [
        "What is a RAG system?",
        "How does Product Quantization work?",
        "What is HNSW and how is it used in vector search?",
        "What are the benefits of reranking in RAG systems?",
        "How are documents split into chunks in a RAG system?"
    ]
    
    benchmark_results = rag.benchmark(benchmark_questions)
    print("\nBenchmark Results:")
    print(benchmark_results[["question", "k", "rerank", "total_time", "retrieval_time", "generation_time"]])
    
    # Compare with fine-tuned model
    comparison_results = compare_with_finetuned(rag, FINETUNED_MODEL_PATH, benchmark_questions)
    print("\nComparison with Fine-tuned Model:")
    print(comparison_results[["question", "rag_time", "ft_time", "time_difference"]])