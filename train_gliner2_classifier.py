"""
GLiNER2 Ad Classifier Training Script

Trains a GLiNER2 model for ad detection using prepared training data.
Uses the official GLiNER2 training API from:
https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md

Training data format (JSONL):
{"input": "text", "output": {"classifications": [{"task": "ad_detection", "labels": ["ad", "no_ad"], "true_label": ["ad"]}]}}
"""

import os
import json
import argparse
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

# GLiNER2 imports - using official API
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, Classification
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


def load_jsonl_dataset(file_path: str) -> List[Dict]:
    """Load JSONL dataset file"""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def convert_to_input_examples(jsonl_data: List[Dict]) -> List[InputExample]:
    """
    Convert JSONL records to GLiNER2 InputExample objects.
    
    JSONL format:
    {
        "input": "transcript text...",
        "output": {
            "classifications": [
                {"task": "ad_detection", "labels": ["ad", "no_ad"], "true_label": ["ad"]}
            ]
        }
    }
    """
    examples = []
    
    for record in jsonl_data:
        text = record.get("input", "")
        output = record.get("output", {})
        classifications_data = output.get("classifications", [])
        
        if not text or not classifications_data:
            continue
        
        # Convert to Classification objects
        classifications = []
        for cls_data in classifications_data:
            classifications.append(
                Classification(
                    task=cls_data.get("task", "ad_detection"),
                    labels=cls_data.get("labels", ["ad", "no_ad"]),
                    true_label=cls_data.get("true_label", [])
                )
            )
        
        # Create InputExample
        examples.append(
            InputExample(
                text=text,
                classifications=classifications
            )
        )
    
    return examples


def train_ad_classifier(
    train_path: str,
    val_path: str,
    output_dir: str = "./gliner2_ad_classifier",
    base_model: str = "fastino/gliner2-base-v1",
    num_epochs: int = 10,
    batch_size: int = 8,
    encoder_lr: float = 1e-5,
    task_lr: float = 5e-4,
    warmup_ratio: float = 0.1,
    fp16: bool = True,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    early_stopping: bool = True,
    early_stopping_patience: int = 3,
    report_to_wandb: bool = False,
    wandb_project: str = "ad_classifier"
) -> str:
    """
    Train GLiNER2 ad classifier using official API.
    
    Args:
        train_path: Path to training JSONL file
        val_path: Path to validation JSONL file
        output_dir: Output directory for model
        base_model: Base GLiNER2 model to fine-tune
        num_epochs: Number of training epochs
        batch_size: Training batch size
        encoder_lr: Learning rate for encoder (pretrained layers)
        task_lr: Learning rate for task head
        warmup_ratio: Warmup ratio for scheduler
        fp16: Use mixed precision training
        use_lora: Use LoRA for parameter-efficient training
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        early_stopping: Enable early stopping
        early_stopping_patience: Patience for early stopping
        report_to_wandb: Report metrics to W&B
        wandb_project: W&B project name
    
    Returns:
        Path to trained model
    """
    print("=" * 70)
    print("🎯 GLiNER2 Ad Classifier Training")
    print("   Using official GLiNER2 training API")
    print("=" * 70)
    
    # Load and convert data
    print(f"\n📊 Loading training data from: {train_path}")
    train_jsonl = load_jsonl_dataset(train_path)
    train_data = convert_to_input_examples(train_jsonl)
    print(f"   Loaded {len(train_data)} training examples")
    
    print(f"\n📊 Loading validation data from: {val_path}")
    val_jsonl = load_jsonl_dataset(val_path)
    val_data = convert_to_input_examples(val_jsonl)
    print(f"   Loaded {len(val_data)} validation examples")
    
    # Count labels
    train_ads = sum(1 for ex in train_data 
                   for cls in ex.classifications 
                   if "ad" in cls.true_label)
    train_no_ads = len(train_data) - train_ads
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Training - Ads: {train_ads}, Non-ads: {train_no_ads}")
    
    # Load base model
    print(f"\n📦 Loading base model: {base_model}")
    model = GLiNER2.from_pretrained(base_model)
    
    # Create training configuration
    print(f"\n⚙️ Training Configuration:")
    print(f"   Output dir: {output_dir}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Encoder LR: {encoder_lr}")
    print(f"   Task LR: {task_lr}")
    print(f"   Warmup ratio: {warmup_ratio}")
    print(f"   FP16: {fp16}")
    print(f"   LoRA: {use_lora}")
    if use_lora:
        print(f"   LoRA rank: {lora_r}, alpha: {lora_alpha}")
    print(f"   Early stopping: {early_stopping} (patience: {early_stopping_patience})")
    
    # Build config
    config = TrainingConfig(
        output_dir=output_dir,
        experiment_name="ad_classifier",
        num_epochs=num_epochs,
        batch_size=batch_size,
        encoder_lr=encoder_lr,
        task_lr=task_lr,
        warmup_ratio=warmup_ratio,
        scheduler_type="cosine",
        fp16=fp16,
        eval_strategy="epoch",  # Evaluate at end of each epoch
        save_best=True,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        # LoRA settings
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        # Logging
        report_to_wandb=report_to_wandb,
        wandb_project=wandb_project if report_to_wandb else None,
    )
    
    # Create trainer
    trainer = GLiNER2Trainer(model, config)
    
    # Train
    print("\n🚀 Starting training...")
    print("-" * 70)
    start_time = datetime.now()
    
    results = trainer.train(
        train_data=train_data,
        eval_data=val_data
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Print results
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"   Duration: {duration}")
    print(f"   Total steps: {results.get('total_steps', 'N/A')}")
    print(f"   Best validation loss: {results.get('best_metric', 'N/A')}")
    
    if 'total_time_seconds' in results:
        print(f"   Training time: {results['total_time_seconds']/60:.1f} minutes")
    
    # Best model path
    best_model_path = os.path.join(output_dir, "best")
    print(f"\n   Best model saved to: {best_model_path}")
    
    return best_model_path


def train_with_jsonl_directly(
    train_path: str,
    val_path: str,
    output_dir: str = "./gliner2_ad_classifier",
    base_model: str = "fastino/gliner2-base-v1",
    num_epochs: int = 10,
    batch_size: int = 8,
    encoder_lr: float = 1e-5,
    task_lr: float = 5e-4,
) -> str:
    """
    Alternative: Train directly from JSONL file.
    GLiNER2 can read JSONL directly without conversion.
    """
    print("=" * 70)
    print("🎯 GLiNER2 Ad Classifier Training (JSONL Mode)")
    print("=" * 70)
    
    print(f"\n📦 Loading base model: {base_model}")
    model = GLiNER2.from_pretrained(base_model)
    
    config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        encoder_lr=encoder_lr,
        task_lr=task_lr,
        warmup_ratio=0.1,
        fp16=True,
        eval_strategy="epoch",
        save_best=True,
    )
    
    print(f"\n⚙️ Config: epochs={num_epochs}, batch={batch_size}, enc_lr={encoder_lr}, task_lr={task_lr}")
    
    trainer = GLiNER2Trainer(model, config)
    
    print(f"\n🚀 Training from JSONL files...")
    print(f"   Train: {train_path}")
    print(f"   Val: {val_path}")
    
    # GLiNER2 can accept file paths directly
    results = trainer.train(
        train_data=train_path,  # Pass JSONL path directly
        eval_data=val_path
    )
    
    print(f"\n✅ Training complete!")
    print(f"   Best model: {output_dir}/best")
    
    return os.path.join(output_dir, "best")


def evaluate_model(model_path: str, test_path: str) -> Dict:
    """
    Evaluate trained model on test data.
    """
    print(f"\n📊 Evaluating model: {model_path}")
    print(f"   Test data: {test_path}")
    
    # Load model
    model = GLiNER2.from_pretrained(model_path)
    
    # Load test data
    test_data = load_jsonl_dataset(test_path)
    
    # Classification labels
    labels = {
        "ad": "Sponsored advertisement or paid promotion",
        "no_ad": "Regular content, not a sponsored advertisement"
    }
    
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    
    print(f"\n🔍 Running predictions on {len(test_data)} examples...")
    
    # Create classification schema once
    schema = model.create_schema().classification(
        "ad_detection",
        ["ad", "no_ad"]
    )
    
    for i, record in enumerate(test_data):
        text = record['input']
        true_labels = record['output']['classifications'][0]['true_label']
        true_label = true_labels[0] if true_labels else "no_ad"
        
        try:
            # Classify using schema-based extraction
            result = model.extract(text, schema, include_confidence=True)
            
            if isinstance(result, dict):
                ad_detection = result.get("ad_detection", {})
                
                if isinstance(ad_detection, dict):
                    pred_label = ad_detection.get("label", "no_ad")
                else:
                    pred_label = str(ad_detection) if ad_detection else "no_ad"
            else:
                pred_label = str(result) if result else "no_ad"
        except Exception as e:
            print(f"  ⚠️ Error on example {i}: {e}")
            pred_label = "no_ad"
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        if pred_label == true_label:
            correct += 1
        total += 1
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(test_data)}...")
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    # Confusion matrix
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "ad" and p == "ad")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "no_ad" and p == "ad")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "ad" and p == "no_ad")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "no_ad" and p == "no_ad")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "=" * 70)
    print("📈 Evaluation Results")
    print("=" * 70)
    
    print(f"\n📊 Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    ad      no_ad")
    print(f"   Actual ad       {tp:5d}    {fn:5d}")
    print(f"          no_ad    {fp:5d}    {tn:5d}")
    
    print(f"\n📊 Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({correct}/{total})")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train GLiNER2 ad classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_gliner2_classifier.py
  
  # With custom settings
  python train_gliner2_classifier.py --epochs 15 --batch-size 16
  
  # Using LoRA for parameter-efficient training
  python train_gliner2_classifier.py --use-lora --lora-r 16
  
  # Evaluate existing model
  python train_gliner2_classifier.py --eval-only ./gliner2_ad_classifier/best
        """
    )
    
    # Data paths
    parser.add_argument('--train-data', type=str, default='./training_data/train_latest.jsonl',
                        help='Path to training data JSONL')
    parser.add_argument('--val-data', type=str, default='./training_data/val_latest.jsonl',
                        help='Path to validation data JSONL')
    
    # Model settings
    parser.add_argument('--base-model', type=str, default='fastino/gliner2-base-v1',
                        help='Base GLiNER2 model to fine-tune')
    parser.add_argument('--output-dir', type=str, default='./gliner2_ad_classifier',
                        help='Output directory for trained model')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, 
                        help='Training batch size')
    parser.add_argument('--encoder-lr', type=float, default=1e-5, 
                        help='Encoder learning rate')
    parser.add_argument('--task-lr', type=float, default=5e-4, 
                        help='Task head learning rate')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, 
                        help='Warmup ratio')
    
    # LoRA settings
    parser.add_argument('--use-lora', action='store_true',
                        help='Use LoRA for parameter-efficient training')
    parser.add_argument('--lora-r', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=float, default=16.0,
                        help='LoRA alpha scaling')
    
    # Other settings
    parser.add_argument('--no-fp16', action='store_true', 
                        help='Disable FP16 training')
    parser.add_argument('--no-early-stopping', action='store_true',
                        help='Disable early stopping')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--wandb', action='store_true',
                        help='Report to Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='ad_classifier',
                        help='W&B project name')
    
    # Modes
    parser.add_argument('--jsonl-mode', action='store_true',
                        help='Train directly from JSONL (simpler)')
    parser.add_argument('--eval-only', type=str, default=None,
                        help='Evaluate an existing model instead of training')
    
    args = parser.parse_args()
    
    # Evaluation mode
    if args.eval_only:
        evaluate_model(args.eval_only, args.val_data)
        return
    
    # Check data exists
    if not os.path.exists(args.train_data):
        print(f"❌ Training data not found: {args.train_data}")
        print("   Run: python prepare_training_data.py")
        return
    
    if not os.path.exists(args.val_data):
        print(f"❌ Validation data not found: {args.val_data}")
        print("   Run: python prepare_training_data.py")
        return
    
    # Train
    if args.jsonl_mode:
        model_path = train_with_jsonl_directly(
            train_path=args.train_data,
            val_path=args.val_data,
            output_dir=args.output_dir,
            base_model=args.base_model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            encoder_lr=args.encoder_lr,
            task_lr=args.task_lr,
        )
    else:
        model_path = train_ad_classifier(
            train_path=args.train_data,
            val_path=args.val_data,
            output_dir=args.output_dir,
            base_model=args.base_model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            encoder_lr=args.encoder_lr,
            task_lr=args.task_lr,
            warmup_ratio=args.warmup_ratio,
            fp16=not args.no_fp16,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            early_stopping=not args.no_early_stopping,
            early_stopping_patience=args.patience,
            report_to_wandb=args.wandb,
            wandb_project=args.wandb_project,
        )
    
    # Evaluate the trained model
    print("\n" + "=" * 70)
    print("📊 Final Evaluation on Validation Set")
    print("=" * 70)
    evaluate_model(model_path, args.val_data)


if __name__ == "__main__":
    main()
