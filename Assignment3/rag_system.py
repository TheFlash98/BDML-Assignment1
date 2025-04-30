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
from tqdm import tqdm  # For better display in notebooks/Colab

# Configure logging for better display in Colab
import sys
import argparse
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Embedding model loaded and moved to {self.device}")
    
    def get_embedding(self, text: str) -> np.ndarray:
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
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            attention_mask = encoded_input['attention_mask']
            token_embeddings = model_output[0]  # First element contains token embeddings
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
            batch_embeddings = normalize(batch_embeddings, norm='l2', axis=1)
            
            embeddings.append(batch_embeddings)
            
        all_embeddings = np.vstack(embeddings)
        
        return all_embeddings

class DocumentRetriever:
    
    def __init__(self, index_path: str, metadata_path: str, embedding_model: EmbeddingModel):
        logger.info(f"Loading index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        logger.info(f"Loading metadata from {metadata_path}")
        self.metadata = self._load_metadata(metadata_path)
        
        self.embedding_model = embedding_model
    
    def _load_metadata(self, metadata_path: str) -> List[Dict]:
        metadata = []
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    
    def search(self, query: str, k: int = 5, rerank: bool = False) -> Tuple[List[Dict], List[float]]:
        logger.info(f"Searching for: {query}")
        
        query_embedding = self.embedding_model.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding, k=k)
        
        chunks = []
        similarities = []
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:
                similarity = 1.0 - score / 2.0
                similarities.append(similarity)
                chunks.append(self.metadata[idx])
                logger.info(f"Retrieved chunk {i+1}: Similarity={similarity:.4f}, Source={self.metadata[idx]['metadata'].get('source', 'unknown')}")
        
        if rerank and chunks:
            chunks, similarities = self._rerank(query, chunks, similarities)
        
        return chunks, similarities
    
    def _rerank(self, query: str, chunks: List[Dict], similarities: List[float]) -> Tuple[List[Dict], List[float]]:
        logger.info("Applying reranking")
        reranked_items = []
        
        query_terms = set(query.lower().split())
        
        for chunk, similarity in zip(chunks, similarities):
            text = chunk["text"].lower()
            
            chunk_terms = set(text.split())
            overlap = len(query_terms.intersection(chunk_terms))
            
            adjusted_score = similarity * (1 + 0.1 * overlap)
            
            reranked_items.append((chunk, adjusted_score))
        
        reranked_items.sort(key=lambda x: x[1], reverse=True)
        
        reranked_chunks, reranked_similarities = zip(*reranked_items) if reranked_items else ([], [])
        
        return list(reranked_chunks), list(reranked_similarities)

class LLaMAGenerator:
    
    def __init__(self, model_path: str = "/scratch/BDML25SP/llama-3b"):
        logger.info(f"Loading LLaMA model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        load_config = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        
        if device == "cuda":
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
                temperature: float = 0.1,
                top_p: float = 0.85,
                repetition_penalty: float = 1.15) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
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
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        end_time = time.time()
        logger.info(f"Generation completed in {end_time - start_time:.2f} seconds")
        
        return generated_text

class RAGSystem:
    
    def __init__(self, 
                index_path: str, 
                metadata_path: str, 
                embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                llama_model_path: str = "/scratch/BDML25SP/llama-3b"):
        self.embedding_model = EmbeddingModel(embedding_model_name)
        
        self.retriever = DocumentRetriever(index_path, metadata_path, self.embedding_model)
        
        self.generator = LLaMAGenerator(llama_model_path)
    
    def answer_question(self, 
                        question: str, 
                        k: int = 5, 
                        rerank: bool = True,
                        show_retrieved: bool = False) -> Dict:
        logger.info(f"Processing question: {question}")
        
        start_time = time.time()
        
        retrieval_start = time.time()
        chunks, similarities = self.retriever.search(question, k=k, rerank=rerank)
        retrieval_time = time.time() - retrieval_start
        
        context_texts = [f"[Document {i+1}]: {chunk['text']}" for i, chunk in enumerate(chunks)]
        context = "\n\n".join(context_texts)
        
        prompt = self._create_prompt(question, context)
        
        logger.info(f"Generated prompt for LLM:\n{prompt}")
        generation_start = time.time()
        answer = self.generator.generate(prompt)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        logger.info(f"Answer generated: {answer}")
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
        return f"""You are a helpful assistant. Answer the question based on the provided context.
If the context doesn't contain relevant information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

    def benchmark(self, questions: List[str], k_values: List[int] = [3, 5, 7]) -> pd.DataFrame:
        results = []
        
        for question in tqdm(questions, desc="Benchmarking questions"):
            for k in k_values:
                for rerank in [False, True]:
                    # Run RAG with current settings
                    result = self.answer_question(question, k=k, rerank=rerank)
                    
                    # Calculate time per token
                    total_tokens = len(result["answer"].split())
                    time_per_token = result["performance"]["total_time"] / total_tokens if total_tokens > 0 else 0
                    
                    # Record result
                    results.append({
                        "question": question,
                        "k": k,
                        "rerank": rerank,
                        "answer": result["answer"],
                        "total_time": result["performance"]["total_time"],
                        "retrieval_time": result["performance"]["retrieval_time"],
                        "generation_time": result["performance"]["generation_time"],
                        "time_per_token": time_per_token,
                    })
        
        return pd.DataFrame(results)

def compare_with_finetuned(rag_system: RAGSystem, finetuned_model_path: str, questions: List[str]) -> pd.DataFrame:
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
        
        rag_tokens = len(rag_result["answer"].split())
        ft_tokens = len(ft_answer.split())
        
        rag_time_per_token = rag_time / rag_tokens if rag_tokens > 0 else 0
        ft_time_per_token = ft_time / ft_tokens if ft_tokens > 0 else 0
        
        results.append({
            "question": question,
            "rag_answer": rag_result["answer"],
            "ft_answer": ft_answer,
            "rag_time": rag_time,
            "ft_time": ft_time,
            "time_difference": rag_time - ft_time,
            "rag_time_per_token": rag_time_per_token,
            "ft_time_per_token": ft_time_per_token,
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG System")
    parser.add_argument("--index_path", type=str, required=True, help="Path to the FAISS index")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the chunk metadata (JSON or JSONL)")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Name of the embedding model")
    parser.add_argument("--llama_model_path", type=str, required=True, help="Path to the LLaMA model")
    parser.add_argument("--finetuned_model_path", type=str, required=True, help="Path to the fine-tuned LLaMA model")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    args = parser.parse_args()

    INDEX_PATH = args.index_path
    METADATA_PATH = args.metadata_path
    EMBEDDING_MODEL = args.embedding_model
    LLAMA_MODEL_PATH = args.llama_model_path
    FINETUNED_MODEL_PATH = args.finetuned_model_path
    BENCHMARK = args.benchmark
    
    rag = RAGSystem(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        embedding_model_name=EMBEDDING_MODEL,
        llama_model_path=LLAMA_MODEL_PATH
    )
    
    question = "What is the climate?"
    result = rag.answer_question(question, show_retrieved=True)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Total time: {result['performance']['total_time']:.2f} seconds")
    
    if BENCHMARK:
        benchmark_questions = [
            "How has the average global temperature changed over the last 100 years?",
            "What are the top 5 regions most affected by climate change today?",
            "Which sectors contribute the most to global CO₂ emissions?",
            "What is the current atmospheric concentration of CO₂, and how does it compare to pre-industrial levels?",
            "Has the frequency of extreme weather events (e.g., heatwaves, hurricanes) increased in the last 50 years?",
            "Which areas are projected to face the highest risk of sea level rise by 2050?",
            "How is climate change affecting global biodiversity and endangered species?",
            "What are the projected temperature and precipitation changes for my region in the next 20 years?",
            "What will happen if global warming exceeds 2°C above pre-industrial levels?",
            "Which countries are on track to meet their Paris Agreement targets?"
        ]
        embedding_model_name_str = EMBEDDING_MODEL.split("/")[-1]
        index_name = INDEX_PATH.split("/")[0]
        benchmark_results = rag.benchmark(benchmark_questions)
        print("\nBenchmark Results:")
        print(benchmark_results)
        benchmark_file_name = f'benchmark_results_{index_name}_{embedding_model_name_str}.csv'
        benchmark_results.to_csv(benchmark_file_name, index=False)
        print("\nBenchmark Results saved to %s" % benchmark_file_name)
        
        comparison_results = compare_with_finetuned(rag, FINETUNED_MODEL_PATH, benchmark_questions)
        print("\nComparison with Fine-tuned Model:")
        print(comparison_results)
        comparison_file_name = f'comparison_results_{index_name}_{embedding_model_name_str}.csv'
        comparison_results.to_csv(comparison_file_name, index=False)
        print("\nComparison with Fine-tuned Model saved to %s" % comparison_file_name)
        print("\nAll tasks completed.")