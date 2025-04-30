#!/bin/bash
#SBATCH --job-name=flatrag
#SBATCH --account=csci_ga_3033_077-2025sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue


singularity exec --bind /scratch --nv --overlay  /scratch/sk12184/overlay-25GB-500K.ext3:ro /scratch/sk12184/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate bdml_env
cd projects/BDML-Assignment1/Assignment3
python rag_system.py --index_path rag_database/faiss_index_flat.bin --metadata_path rag_database/chunks_metadata_flat.json --embedding_model "sentence-transformers/all-MiniLM-L6-v2" --llama_model_path /scratch/sk12184/llama3.2-3B-HF --finetuned_model_path /scratch/sk12184/output/checkpoint-652 --benchmark"
"