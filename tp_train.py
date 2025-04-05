import os

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from dataset import ClimateDataset
from transformers import DataCollatorWithPadding
import torch
import argparse

import torch.distributed as dist
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel,
    SequenceParallel, parallelize_module, PrepareModuleInput
)
from torch.distributed.device_mesh import init_device_mesh

# torchrun --nproc_per_node=2 tp_train.py  --per_device_train_batch_size 8 --fine_tuning_type "qlora" --use_fp16 --gradient_checkpointing --num_train_epochs 2 --output_dir /scratch/sk12184/output/tensor_parallel/


def setup_tensor_parallel():
    """Initialize distributed environment for tensor parallelism"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"Rank: {rank}, World Size: {world_size}")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    return rank, world_size

def main():
    rank, world_size = setup_tensor_parallel()
    tp_mesh = init_device_mesh("cuda", (world_size,))
    print(f"Rank {rank}: Created device mesh -> {tp_mesh}")
    model_name = "/scratch/sk12184/llama3.2-3B-HF"
    
    train_dataset = ClimateDataset(data_root_path="/scratch/sk12184/climate_text_dataset_tokenized", split="train")
    eval_dataset = ClimateDataset(data_root_path="/scratch/sk12184/climate_text_dataset_tokenized", split="eval")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        fp16 = args.use_fp16,
        bf16 = args.use_bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=4,
        logging_steps=100,
        save_strategy="epoch",
        save_steps=500,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        local_rank=rank,
        ddp_backend=None,
        accelerator_config={
            "distributed_type": "TP",
        }
    )

    tokenizer, model = get_model(args, model_name)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,  # Ensure padding is enabled
    )
    # parallelize_plan = {
    #     "base_model.model.model.embed_tokens": ColwiseParallel(),
    #     # Final LayerNorm - replicated on all devices
    #     "base_model.model.model.norm": SequenceParallel()
    # }
    # for i in range(28):
    #     parallelize_plan.update({
    #         f"base_model.model.model.layers.{i}.self_attn.q_proj.base_layer": ColwiseParallel(),
    #         f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.default": ColwiseParallel(),
    #         f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.default": RowwiseParallel(),
            
    #         f"base_model.model.model.layers.{i}.self_attn.k_proj": ColwiseParallel(),
            
    #         f"base_model.model.model.layers.{i}.self_attn.v_proj.base_layer": ColwiseParallel(),
    #         f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.default": ColwiseParallel(),
    #         f"base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.default": RowwiseParallel(),
            
    #         f"base_model.model.model.layers.{i}.self_attn.o_proj": RowwiseParallel(),
            
    #         f"base_model.model.model.layers.{i}.mlp.gate_proj": ColwiseParallel(),
    #         f"base_model.model.model.layers.{i}.mlp.up_proj": ColwiseParallel(),
    #         f"base_model.model.model.layers.{i}.mlp.down_proj": RowwiseParallel(),
            
    #         f"base_model.model.model.layers.{i}.input_layernorm": SequenceParallel(),
    #         f"base_model.model.model.layers.{i}.post_attention_layernorm": SequenceParallel()
    #     })
    model = model.to("cuda")
    if dist.get_rank() == 0:
        print("\nModel structure BEFORE parallelization:")
        for name, param in model.named_parameters():
             print(f"{name:80} | Device: {param.device} | Shape: {param.shape}")
    
    
    parallelize_module(
        model.model,
        tp_mesh,
        {
            "embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
                use_local_output=False,
            ),
        },
    )
    rowwise_parallel, colwise_parallel, prepare_module_input = (
        RowwiseParallel,
        ColwiseParallel,
        PrepareModuleInput,
    )
    for transformer_block in model.model.layers:
        layer_plan = {
            "self_attn.q_proj": colwise_parallel(),
            "self_attn.k_proj": colwise_parallel(),
            "self_attn.v_proj": colwise_parallel(),
            "self_attn.o_proj": rowwise_parallel(),
            "mlp.gate_proj": colwise_parallel(),
            "mlp.up_proj": colwise_parallel(),
            "mlp.down_proj": rowwise_parallel(),
            "input_layernorm": SequenceParallel(),
            "post_attention_layernorm": SequenceParallel(),
        }
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )
    
    print(f"\nRank {dist.get_rank()} parameters:")
    for name, param in model.named_parameters():
        print(f"{name:80} | Device: {param.device} | Shape: {param.shape}")
    
    #print(list(model.named_parameters()))
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()

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
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
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

