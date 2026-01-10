#!/usr/bin/env python3
"""
Fine-tune microsoft/deberta-v3-base for Ad Detection in Podcast Transcripts

This script trains a DeBERTa-v3-base model for binary classification to detect
advertisements in podcast audio transcripts.

Usage:
    python train_deberta.py --config config.json
    python train_deberta.py  # Uses default configuration
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_CONFIG = {
    "model_name": "microsoft/deberta-v3-base",
    "train_file": "training_data/train_latest.jsonl",
    "val_file": "training_data/val_latest.jsonl",
    "output_dir": "deberta_ad_classifier/output",
    "max_length": 512,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "evaluation_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 50,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    "fp16": torch.cuda.is_available(),
    "seed": 42,
    "early_stopping_patience": 3,
    "logging_steps": 10,
    "report_to": "none",
}


def load_jsonl_data(file_path: str) -> list:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num} due to JSON error: {e}")
    return data


def parse_data_item(item: dict) -> Optional[Dict]:
    """Parse a single data item into text and label."""
    try:
        text = item.get("input", "")
        output = item.get("output", {})
        
        # Extract label from the nested structure
        classifications = output.get("classifications", [])
        if not classifications:
            return None
        
        # Get the first classification (ad_detection task)
        classification = classifications[0]
        true_label = classification.get("true_label", [])
        
        if isinstance(true_label, list):
            label_str = true_label[0] if true_label else None
        else:
            label_str = true_label
        
        if label_str is None:
            return None
        
        # Convert to binary label: ad=1, no_ad=0
        label = 1 if label_str == "ad" else 0
        
        return {"text": text, "label": label}
    except Exception as e:
        logger.warning(f"Error parsing item: {e}")
        return None


def prepare_dataset(train_file: str, val_file: str) -> DatasetDict:
    """Load and prepare the dataset."""
    logger.info(f"Loading training data from: {train_file}")
    train_raw = load_jsonl_data(train_file)
    
    logger.info(f"Loading validation data from: {val_file}")
    val_raw = load_jsonl_data(val_file)
    
    # Parse data
    train_data = [parse_data_item(item) for item in train_raw]
    train_data = [d for d in train_data if d is not None]
    
    val_data = [parse_data_item(item) for item in val_raw]
    val_data = [d for d in val_data if d is not None]
    
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    
    # Log class distribution
    train_ads = sum(1 for d in train_data if d["label"] == 1)
    train_no_ads = len(train_data) - train_ads
    val_ads = sum(1 for d in val_data if d["label"] == 1)
    val_no_ads = len(val_data) - val_ads
    
    logger.info(f"Train - Ads: {train_ads}, No Ads: {train_no_ads}")
    logger.info(f"Val - Ads: {val_ads}, No Ads: {val_no_ads}")
    
    # Create Hugging Face datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return DatasetDict({"train": train_dataset, "validation": val_dataset})


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="binary", zero_division=0)
    recall = recall_score(labels, predictions, average="binary", zero_division=0)
    f1 = f1_score(labels, predictions, average="binary", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the input text."""
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=max_length,
    )


def load_config(config_path: Optional[str]) -> dict:
    """Load configuration from file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, "r") as f:
            user_config = json.load(f)
        config.update(user_config)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for ad detection")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--model_name", type=str, help="Model name or path")
    parser.add_argument("--train_file", type=str, help="Training data file")
    parser.add_argument("--val_file", type=str, help="Validation data file")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_name:
        config["model_name"] = args.model_name
    if args.train_file:
        config["train_file"] = args.train_file
    if args.val_file:
        config["val_file"] = args.val_file
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.epochs:
        config["num_train_epochs"] = args.epochs
    if args.batch_size:
        config["per_device_train_batch_size"] = args.batch_size
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.seed:
        config["seed"] = args.seed
    
    # Set seed for reproducibility
    set_seed(config["seed"])
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config["output_dir"]) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = output_dir / "training_config.json"
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to: {config_save_path}")
    
    logger.info("=" * 60)
    logger.info("DeBERTa Ad Classifier Training")
    logger.info("=" * 60)
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2,
        id2label={0: "no_ad", 1: "ad"},
        label2id={"no_ad": 0, "ad": 1},
    )
    
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    # Prepare dataset
    dataset = prepare_dataset(config["train_file"], config["val_file"])
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        evaluation_strategy=config["evaluation_strategy"],
        eval_steps=config["eval_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        fp16=config["fp16"],
        logging_steps=config["logging_steps"],
        logging_dir=str(output_dir / "logs"),
        report_to=config["report_to"],
        seed=config["seed"],
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )
    
    # Callbacks
    callbacks = []
    if config.get("early_stopping_patience"):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])
        )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save the best model
    best_model_dir = output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    logger.info(f"Best model saved to: {best_model_dir}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    # Detailed predictions for analysis
    predictions = trainer.predict(tokenized_dataset["validation"])
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Classification report
    report = classification_report(
        true_labels, 
        pred_labels, 
        target_names=["no_ad", "ad"],
        digits=4,
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Save results
    results = {
        "train_metrics": {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        },
        "eval_metrics": eval_results,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "config": config,
    }
    
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nTraining Loss: {train_result.training_loss:.4f}")
    logger.info(f"\nEvaluation Results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
    logger.info(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Best model saved to: {best_model_dir}")
    
    return results


if __name__ == "__main__":
    main()
