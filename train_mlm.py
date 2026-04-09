import os
import argparse
import logging
import torch
import time
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
    parser.add_argument("--dataset_name", type=str, default="bigcode/the-stack-v2")
    parser.add_argument("--subset", type=str, default="default", help="Language subfolder (e.g., 'Python', 'Rust')")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str)
    return parser.parse_args()

filter_stats = {"scanned": 0, "accepted": 0, "last_log": time.time()}

def train():
    args = parse_args()
    set_seed(args.seed)
    
    if args.report_to == "wandb":
        import wandb
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
            wandb.init(project="arcanon-codebert-mlm", name=f"phase-1-mlm-{args.subset}")
    
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=token, add_prefix_space=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, token=token)
    model.to(device)

    # SMOKE TEST BYPASS
    is_smoke_test = args.max_steps < 100
    
    # Target specific data directories (e.g. data/Python)
    data_dir = None
    if is_smoke_test:
        data_dir = "data/Python"
    elif args.subset != "default":
        data_dir = f"data/{args.subset}"

    logger.info(f"Connecting to {args.dataset_name} (Data Dir: {data_dir if data_dir else 'All'})")
    
    raw_dataset = load_dataset(
        args.dataset_name, 
        data_dir=data_dir,
        streaming=True, 
        split="train", 
        token=token
    )

    def filter_samples(sample):
        filter_stats["scanned"] += 1
        
        # SMOKE TEST BYPASS: Accept the first valid looking files immediately
        if is_smoke_test:
            if len(str(sample.get("content", ""))) > 100:
                filter_stats["accepted"] += 1
                return True
            return False

        # --- REAL PRODUCTION FILTERING ---
        now = time.time()
        if filter_stats["scanned"] % 1000 == 0 or (now - filter_stats["last_log"]) > 10:
            logger.info(f"Scanned {filter_stats['scanned']} files, Found {filter_stats['accepted']} matches...")
            filter_stats["last_log"] = now

        content = str(sample.get("content", ""))
        lang = str(sample.get("lang", sample.get("language", ""))).lower()
        lic = str(sample.get("license_type", "")).lower()
        
        # Flexible filters
        target_languages = ["rust", "go", "python", "java", "typescript", "javascript", "ruby", "csharp", "c#", "asp.net", "js", "ts"]
        permissive_keywords = ["mit", "apache", "bsd", "isc", "permissive", "public"]
        service_keywords = ["express", "axum", "tokio", "fastapi", "flask", "django", "spring", "controller", "route", "service", "client"]

        if not any(t in lang for t in target_languages) and data_dir is None: return False
        if not any(p in lic for p in permissive_keywords) and lic != "": return False
        if len(content) < 200 or len(content) > 100000: return False
        
        is_service_glue = any(kw in content[:2000].lower() for kw in service_keywords)
        if is_service_glue:
            filter_stats["accepted"] += 1
            return True
        return False

    dataset = raw_dataset.filter(filter_samples)

    def tokenize_function(examples):
        return tokenizer(examples["content"], truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=1 if is_smoke_test else 50,
        fp16=use_fp16,
        report_to=args.report_to,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    logger.info(f"Starting Phase 1 Training for {args.max_steps} steps...")
    trainer.train()

    logger.info(f"Saving weights to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    train()
