#!/bin/bash
#SBATCH --job-name=b8e2
#SBATCH --account=csci_ga_3033_077-2025sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue


singularity exec --bind /scratch --nv --overlay  /scratch/sk12184/overlay-25GB-500K.ext3:rw /scratch/sk12184/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate bdml_env
cd projects/BDML-Assignment1
python train.py --per_device_train_batch_size 8 --fine_tuning_type "qlora" --use_fp16 --gradient_checkpointing --num_train_epochs 2
"