import os

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from dataset import ClimateDataset
from transformers import DataCollatorWithPadding
import torch
import argparse
from tqdm import tqdm
import torch.distributed as dist
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel,
    SequenceParallel, parallelize_module, PrepareModuleInput
)
from torch.optim import AdamW
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import TorchTensorParallelPlugin
from transformers.trainer_pt_utils import AcceleratorConfig
# torchrun --nproc_per_node=2 tp_train.py  --per_device_train_batch_size 8 --fine_tuning_type "qlora" --use_fp16 --gradient_checkpointing --num_train_epochs 2 --output_dir /scratch/sk12184/output/tensor_parallel/

#accelerate launch --config_file ~/.cache/huggingface/accelerate/default_config.yaml --num_processes=2  accelerate_train.py  --per_device_train_batch_size 1 --fine_tuning_type "full" --use_fp16 --gradient_checkpointing --num_train_epochs 2 --output_dir /scratch/sk12184/output/tensor_parallel/ >> debug.txt

def setup_tensor_parallel():
    """Initialize distributed environment for tensor parallelism"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"Rank: {rank}, World Size: {world_size}")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        # init_method="env://",
        # rank=rank,
        # world_size=world_size
    )
    return rank, world_size

def main():
    rank, world_size = setup_tensor_parallel()
    # print(f"Rank {rank}: Created device mesh -> {tp_mesh}")
    model_name = "/scratch/sk12184/llama3.2-3B-HF"
    accelerate = Accelerator(
        mixed_precision="fp16" if args.use_fp16 else "bf16" if args.use_bf16 else None,
        project_dir=args.output_dir,
    )
    train_dataset = ClimateDataset(data_root_path="/scratch/sk12184/climate_text_dataset_tokenized", split="train")
    eval_dataset = ClimateDataset(data_root_path="/scratch/sk12184/climate_text_dataset_tokenized", split="eval")

    
    tokenizer, model = get_model(args, model_name)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,  # Ensure padding is enabled
    )
    # model = model.to("cuda")
    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        # sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    train_loader, model, optimizer = accelerate.prepare(train_loader, model, optimizer)
    model.train()
    for epoch in range(args.num_train_epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.num_train_epochs}")
        for step, batch in pbar:
            inputs = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]

            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            accelerate.backward(loss)
            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        accelerate.save_state(args.output_dir)
        accelerate.load_state(args.output_dir)
    if rank == 0:
        try:
            accelerate.wait_for_everyone()
            accelerate.save_model(model, args.output_dir)
            print(f"Model saved to {args.output_dir}")
        except Exception as e:
            print(f"Error saving model: {e}")

    

def get_model(args, model_name):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        truncation=True,
        padding=True,
        max_length=1730,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    if args.fine_tuning_type == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
        prepare_model_for_kbit_training(base_model, use_gradient_checkpointing = args.gradient_checkpointing)
        if args.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        
        model = get_peft_model(base_model, lora_config)
        # model.print_trainable_parameters()
    elif args.fine_tuning_type == "lora":  
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if args.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        model = get_peft_model(base_model, lora_config)

        # model.print_trainable_parameters()
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
        
        if args.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
        
        model = base_model
    
    return tokenizer, model

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with LoRA")
    parser.add_argument("--output_dir", type=str, default="/scratch/sk12184/output", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--use_fp16", action="store_true", help="Use fp16")
    parser.add_argument("--use_bf16", action="store_true", help="Use bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Per device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--fine_tuning_type", type=str, default="lora",choices=["qlora", "lora","full"], help="Fine tuning type")

    args = parser.parse_args()
    
    if args.use_fp16 and args.use_bf16:
        raise ValueError("Cannot use both fp16 and bf16")
    
    world_size = torch.cuda.device_count()
    print(f"Number of GPUs: {world_size}")
    main()

