#!/bin/bash
#SBATCH --job-name=tp2
#SBATCH --account=csci_ga_3033_077-2025sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:2
#SBATCH --requeue


singularity exec --bind /scratch --nv --overlay  /scratch/sk12184/overlay-25GB-500K.ext3:ro /scratch/sk12184/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate bdml_env
cd projects/BDML-Assignment1
accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml --num_processes=2  accelerate_train.py  --per_device_train_batch_size 1 --fine_tuning_type "full" --use_fp16 --gradient_checkpointing --num_train_epochs 1 --output_dir /scratch/sk12184/output/tensor_parallel/tp-model-1
"