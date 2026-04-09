import os
import argparse
import logging
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding
)
from datasets import load_dataset
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Fine-tune CodeBERT for Service Classification")
    parser.add_argument("--model_name_or_path", type=str, default="./checkpoints")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./classifier_checkpoints")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str)
    return parser.parse_args()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train():
    args = parse_args()
    set_seed(args.seed)
    
    # Explicit WandB Login for automated environments
    if args.report_to == "wandb":
        import wandb
        api_key = os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
            wandb.init(project="arcanon-codebert-classifier", name="phase-2-sbc")
        else:
            logger.warning("WANDB_API_KEY not found. WandB logging may fail.")
    
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, add_prefix_space=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)
    model.to(device)

    data_files = {"train": args.train_file}
    if args.validation_file:
        data_files["validation"] = args.validation_file
    
    raw_datasets = load_dataset("json", data_files=data_files)

    def tokenize_function(examples):
        return tokenizer(examples["content"], truncation=True, max_length=512, padding=False)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["content"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch" if args.validation_file else "no",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        save_steps=args.save_steps,
        fp16=use_fp16,
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=os.getenv("HF_TOKEN"),
        load_best_model_at_end=True if args.validation_file else False,
        metric_for_best_model="f1" if args.validation_file else None,
        dataloader_num_workers=0 if device == "mps" else 4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting Phase 2 Classification training...")
    trainer.train()

    logger.info(f"Saving classifier to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        logger.info(f"Pushing to Hugging Face Hub: {args.hub_model_id}")
        trainer.push_to_hub()

if __name__ == "__main__":
    train()
