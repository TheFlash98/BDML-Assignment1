from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from dataset import ClimateDataset
from transformers import DataCollatorWithPadding
import torch
import argparse
from datasets import load_dataset

def main(args):
    model_name = "/scratch/sk12184/llama3.2-3B-HF"
    
    train_dataset = ClimateDataset(data_root_path="/scratch/sk12184/climate_text_dataset_tokenized", split="train")
    eval_dataset = ClimateDataset(data_root_path="/scratch/sk12184/climate_text_dataset_tokenized", split="eval")
    
    # train_dataset = load_dataset("imdb", split="train")
    # eval_dataset = load_dataset("imdb", split="test")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        fp16 = args.use_fp16,
        bf16 = args.use_bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        truncation=True,
        padding=True,
        max_length=1730,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,  # Ensure padding is enabled
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    if args.fine_tuning_type == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp16",
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
        
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
        model.print_trainable_parameters()
    elif args.fine_tuning_type == "lora":  
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        model = get_peft_model(base_model, lora_config)

        model.print_trainable_parameters()
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if args.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
        
        model = base_model
    
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    
    trainer.train()
    
    return
    
    


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
    
    main(args)

