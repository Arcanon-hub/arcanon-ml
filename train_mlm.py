import os
import argparse
import logging
import torch
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)
from datasets import load_dataset
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT (MLM) on RunPod/M3")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/codebert-base")
    parser.add_argument("--dataset_name", type=str, default="bigcode/the-stack-v2-train-full")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=False, help="Use mixed precision (NVIDIA only)")
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--push_to_hub", action="store_true", help="Push the model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, help="The name of the repository on HF (e.g., 'user/repo')")
    return parser.parse_args()

def train():
    args = parse_args()
    set_seed(args.seed)
    
    # Device detection for M3/NVIDIA/CPU
    if torch.cuda.is_available():
        device = "cuda"
        use_fp16 = args.fp16
    elif torch.backends.mps.is_available():
        device = "mps"
        use_fp16 = False
        logger.info("Apple Silicon detected. Using MPS device.")
    else:
        device = "cpu"
        use_fp16 = False
    
    logger.info(f"Training on device: {device}")

    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not found. Ensure it's in your .env or environment variables.")
    
    logger.info(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=token, add_prefix_space=True)
    
    logger.info(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, token=token)
    model.to(device)

    # Filtering Logic
    target_languages = ["rust", "go", "python", "java", "typescript", "javascript", "ruby", "csharp"]
    permissive_licenses = ["mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"]
    
    service_keywords = [
        "express", "axum", "tokio", "fastapi", "flask", "django", "spring", "controller",
        "route", "@app.", "#[get", "axum::extract", "boto3", "kafka", "redis", "postgres",
        "sqlx", "service", "client", "grpc", "protobuf"
    ]

    logger.info(f"Connecting to {args.dataset_name} (Streaming Mode)")
    raw_dataset = load_dataset(args.dataset_name, streaming=True, split="train", token=token)

    def filter_samples(sample):
        content = str(sample.get("content", ""))
        lang = str(sample.get("lang", "")).lower()
        lic = str(sample.get("license_type", "")).lower()
        
        if lang not in target_languages: return False
        if not any(p in lic for p in permissive_licenses): return False
        if len(content) < 200: return False
        if len(content) > 100000: return False
        
        lines = content.split('\n')
        if any(len(line) > 1000 for line in lines[:100]): return False
        
        secret_patterns = ["api_key", "password", "secret", "token", "auth_"]
        content_lower = content[:2000].lower()
        if any(f"{p} =" in content_lower or f"{p}:" in content_lower for p in secret_patterns):
            return False

        is_service_glue = any(kw in content_lower for kw in service_keywords)
        return is_service_glue

    dataset = raw_dataset.filter(filter_samples)

    def tokenize_function(examples):
        return tokenizer(
            examples["content"], 
            truncation=True, 
            max_length=512, 
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=raw_dataset.column_names
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=max(1, args.max_steps // 100) if args.max_steps < 100 else 50,
        fp16=use_fp16,
        use_mps_device=(device == "mps"),
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=token,
        dataloader_num_workers=0 if device == "mps" else 4, # MPS can be unstable with multiple workers
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    logger.info(f"Starting Phase 1 Training for {args.max_steps} steps...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training interrupted: {e}")
        raise e

    logger.info(f"Saving final weights to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        logger.info(f"Pushing to Hugging Face Hub: {args.hub_model_id}")
        trainer.push_to_hub()

if __name__ == "__main__":
    train()
