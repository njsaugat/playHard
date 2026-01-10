"""
GLiNER2 Ad Classifier Training Script

Trains a GLiNER2 model for ad detection using prepared training data.
Uses the official GLiNER2 training API from:
https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md

Training data format (JSONL):
{"input": "text", "output": {"classifications": [{"task": "ad_detection", "labels": ["ad", "no_ad"], "true_label": ["ad"]}]}}

Model Versioning:
- Each training run creates a new timestamped checkpoint
- Models are stored in: {output_dir}/runs/run_{timestamp}/
- A 'latest' symlink always points to the most recent successful model
- Previous models are preserved for rollback
"""

import os
import json
import argparse
import shutil
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# GLiNER2 imports - using official API
from gliner2 import GLiNER2
from gliner2.training.data import InputExample, Classification
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


@dataclass
class TrainingRunMetadata:
    """Metadata for a training run - saved with each checkpoint."""
    run_id: str
    timestamp: str
    base_model: str
    train_data_path: str
    val_data_path: str
    train_samples: int
    val_samples: int
    num_epochs: int
    batch_size: int
    encoder_lr: float
    task_lr: float
    use_lora: bool
    lora_r: Optional[int] = None
    lora_alpha: Optional[float] = None
    training_duration_seconds: Optional[float] = None
    best_validation_loss: Optional[float] = None
    final_accuracy: Optional[float] = None
    final_f1: Optional[float] = None
    notes: Optional[str] = None


def create_versioned_output_dir(base_output_dir: str, run_name: Optional[str] = None) -> tuple[Path, str]:
    """
    Create a versioned output directory for a new training run.
    
    Structure:
        {base_output_dir}/
            runs/
                run_20260110_143052/
                run_20260110_160215/
                ...
            latest -> runs/run_20260110_160215  (symlink)
    
    Returns:
        tuple: (run_dir_path, run_id)
    """
    base_path = Path(base_output_dir)
    runs_dir = base_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate run ID with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        run_id = f"run_{timestamp}_{run_name}"
    else:
        run_id = f"run_{timestamp}"
    
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir, run_id


def update_latest_symlink(base_output_dir: str, run_id: str):
    """
    Update the 'latest' symlink to point to the most recent successful run.
    """
    base_path = Path(base_output_dir)
    latest_link = base_path / "latest"
    run_dir = base_path / "runs" / run_id
    
    # Remove existing symlink if it exists
    if latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        # If it's a regular directory (not symlink), rename it as backup
        backup_name = f"latest_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(latest_link), str(base_path / backup_name))
        print(f"⚠️  Backed up existing 'latest' directory to: {backup_name}")
    
    # Create relative symlink for portability
    relative_target = Path("runs") / run_id
    latest_link.symlink_to(relative_target)
    print(f"✅ Updated 'latest' symlink -> {relative_target}")


def list_available_checkpoints(base_output_dir: str) -> List[Dict]:
    """
    List all available model checkpoints with their metadata.
    
    Returns:
        List of checkpoint info dicts sorted by date (newest first)
    """
    base_path = Path(base_output_dir)
    runs_dir = base_path / "runs"
    
    if not runs_dir.exists():
        return []
    
    checkpoints = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            metadata_file = run_dir / "training_metadata.json"
            model_dir = run_dir / "best"
            
            checkpoint_info = {
                "run_id": run_dir.name,
                "path": str(run_dir),
                "model_path": str(model_dir) if model_dir.exists() else None,
                "has_model": model_dir.exists(),
                "created": run_dir.stat().st_mtime,
            }
            
            # Load metadata if available
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    checkpoint_info["metadata"] = metadata
                except Exception:
                    pass
            
            checkpoints.append(checkpoint_info)
    
    # Sort by creation time (newest first)
    checkpoints.sort(key=lambda x: x["created"], reverse=True)
    return checkpoints


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
    wandb_project: str = "ad_classifier",
    run_name: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Train GLiNER2 ad classifier using official API.
    
    Each training run creates a new versioned checkpoint that is preserved.
    Previous models are NOT overwritten.
    
    Directory structure:
        {output_dir}/
            runs/
                run_20260110_143052/
                    best/           <- trained model
                    training_metadata.json
                    training_config.json
                run_20260110_160215/
                    ...
            latest -> runs/run_20260110_160215  (symlink to most recent)
    
    Args:
        train_path: Path to training JSONL file
        val_path: Path to validation JSONL file
        output_dir: Base output directory for all models
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
        run_name: Optional custom name suffix for this run
        notes: Optional notes about this training run
    
    Returns:
        Path to trained model (the 'best' subdirectory of the run)
    """
    print("=" * 70)
    print("🎯 GLiNER2 Ad Classifier Training")
    print("   Using official GLiNER2 training API")
    print("   📁 With versioned checkpoints (previous models preserved)")
    print("=" * 70)
    
    # Create versioned output directory
    run_dir, run_id = create_versioned_output_dir(output_dir, run_name)
    print(f"\n📂 Created new training run: {run_id}")
    print(f"   Run directory: {run_dir}")
    
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
    print(f"   Run ID: {run_id}")
    print(f"   Output dir: {run_dir}")
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
    
    # Build config - use the run directory as output
    config = TrainingConfig(
        output_dir=str(run_dir),
        experiment_name=f"ad_classifier_{run_id}",
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
    
    # Save training config for reproducibility
    config_save_path = run_dir / "training_config.json"
    with open(config_save_path, 'w') as f:
        json.dump({
            "base_model": base_model,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "encoder_lr": encoder_lr,
            "task_lr": task_lr,
            "warmup_ratio": warmup_ratio,
            "fp16": fp16,
            "use_lora": use_lora,
            "lora_r": lora_r if use_lora else None,
            "lora_alpha": lora_alpha if use_lora else None,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
        }, f, indent=2)
    print(f"   Config saved to: {config_save_path}")
    
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
    duration_seconds = duration.total_seconds()
    
    # Print results
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"   Run ID: {run_id}")
    print(f"   Duration: {duration}")
    print(f"   Total steps: {results.get('total_steps', 'N/A')}")
    print(f"   Best validation loss: {results.get('best_metric', 'N/A')}")
    
    if 'total_time_seconds' in results:
        print(f"   Training time: {results['total_time_seconds']/60:.1f} minutes")
    
    # Best model path (inside the run directory)
    best_model_path = run_dir / "best"
    print(f"\n   Best model saved to: {best_model_path}")
    
    # Save training metadata
    metadata = TrainingRunMetadata(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        base_model=base_model,
        train_data_path=str(Path(train_path).resolve()),
        val_data_path=str(Path(val_path).resolve()),
        train_samples=len(train_data),
        val_samples=len(val_data),
        num_epochs=num_epochs,
        batch_size=batch_size,
        encoder_lr=encoder_lr,
        task_lr=task_lr,
        use_lora=use_lora,
        lora_r=lora_r if use_lora else None,
        lora_alpha=lora_alpha if use_lora else None,
        training_duration_seconds=duration_seconds,
        best_validation_loss=results.get('best_metric'),
        notes=notes,
    )
    
    metadata_path = run_dir / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(asdict(metadata), f, indent=2)
    print(f"   Metadata saved to: {metadata_path}")
    
    # Update 'latest' symlink to point to this run
    update_latest_symlink(output_dir, run_id)
    
    # Show available checkpoints
    print("\n" + "-" * 70)
    print("📋 Available Model Checkpoints:")
    checkpoints = list_available_checkpoints(output_dir)
    for i, cp in enumerate(checkpoints[:5]):  # Show last 5
        is_latest = "(latest)" if i == 0 else ""
        has_model = "✓" if cp["has_model"] else "✗"
        print(f"   [{has_model}] {cp['run_id']} {is_latest}")
    if len(checkpoints) > 5:
        print(f"   ... and {len(checkpoints) - 5} more")
    
    return str(best_model_path)


def train_with_jsonl_directly(
    train_path: str,
    val_path: str,
    output_dir: str = "./gliner2_ad_classifier",
    base_model: str = "fastino/gliner2-base-v1",
    num_epochs: int = 10,
    batch_size: int = 8,
    encoder_lr: float = 1e-5,
    task_lr: float = 5e-4,
    run_name: Optional[str] = None,
) -> str:
    """
    Alternative: Train directly from JSONL file.
    GLiNER2 can read JSONL directly without conversion.
    
    Also uses versioned checkpoints - each run is preserved.
    """
    print("=" * 70)
    print("🎯 GLiNER2 Ad Classifier Training (JSONL Mode)")
    print("   📁 With versioned checkpoints (previous models preserved)")
    print("=" * 70)
    
    # Create versioned output directory
    run_dir, run_id = create_versioned_output_dir(output_dir, run_name)
    print(f"\n📂 Created new training run: {run_id}")
    print(f"   Run directory: {run_dir}")
    
    print(f"\n📦 Loading base model: {base_model}")
    model = GLiNER2.from_pretrained(base_model)
    
    config = TrainingConfig(
        output_dir=str(run_dir),
        experiment_name=f"ad_classifier_{run_id}",
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
    
    start_time = datetime.now()
    
    # GLiNER2 can accept file paths directly
    results = trainer.train(
        train_data=train_path,  # Pass JSONL path directly
        eval_data=val_path
    )
    
    duration = datetime.now() - start_time
    best_model_path = run_dir / "best"
    
    print(f"\n✅ Training complete!")
    print(f"   Run ID: {run_id}")
    print(f"   Duration: {duration}")
    print(f"   Best model: {best_model_path}")
    
    # Update 'latest' symlink
    update_latest_symlink(output_dir, run_id)
    
    return str(best_model_path)


def get_latest_model_path(output_dir: str = "./gliner2_ad_classifier") -> Optional[str]:
    """
    Get the path to the latest trained model.
    
    Returns:
        Path to the latest model's 'best' directory, or None if no models exist.
    """
    base_path = Path(output_dir)
    latest_link = base_path / "latest"
    
    if latest_link.exists():
        best_model = latest_link / "best"
        if best_model.exists():
            return str(best_model.resolve())
    
    # Fallback: find most recent run
    checkpoints = list_available_checkpoints(output_dir)
    for cp in checkpoints:
        if cp["has_model"]:
            return cp["model_path"]
    
    return None


def get_model_by_run_id(output_dir: str, run_id: str) -> Optional[str]:
    """
    Get the path to a specific model checkpoint by run ID.
    
    Args:
        output_dir: Base output directory
        run_id: Run ID (e.g., 'run_20260110_143052')
    
    Returns:
        Path to the model's 'best' directory, or None if not found.
    """
    base_path = Path(output_dir)
    run_dir = base_path / "runs" / run_id / "best"
    
    if run_dir.exists():
        return str(run_dir)
    return None


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


def print_checkpoints(output_dir: str):
    """Print all available checkpoints with their details."""
    print("=" * 70)
    print("📋 Available Model Checkpoints")
    print("=" * 70)
    
    checkpoints = list_available_checkpoints(output_dir)
    
    if not checkpoints:
        print(f"\n   No checkpoints found in: {output_dir}")
        print("   Train a model first with: python train_gliner2_classifier.py")
        return
    
    latest_path = get_latest_model_path(output_dir)
    
    for i, cp in enumerate(checkpoints):
        is_latest = cp["model_path"] == latest_path if latest_path else False
        has_model = "✓" if cp["has_model"] else "✗"
        latest_tag = " ← latest" if is_latest else ""
        
        print(f"\n[{has_model}] {cp['run_id']}{latest_tag}")
        print(f"    Path: {cp['path']}")
        
        if "metadata" in cp:
            meta = cp["metadata"]
            if meta.get("timestamp"):
                print(f"    Trained: {meta['timestamp']}")
            if meta.get("base_model"):
                print(f"    Base model: {meta['base_model']}")
            if meta.get("train_samples"):
                print(f"    Training samples: {meta['train_samples']}")
            if meta.get("best_validation_loss"):
                print(f"    Best val loss: {meta['best_validation_loss']:.4f}")
            if meta.get("training_duration_seconds"):
                mins = meta['training_duration_seconds'] / 60
                print(f"    Training time: {mins:.1f} minutes")
            if meta.get("notes"):
                print(f"    Notes: {meta['notes']}")
    
    print("\n" + "-" * 70)
    print(f"Total checkpoints: {len(checkpoints)}")
    if latest_path:
        print(f"Latest model: {latest_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train GLiNER2 ad classifier with versioned checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (creates new checkpoint, preserves previous models)
  python train_gliner2_classifier.py
  
  # Training with custom name for easy identification
  python train_gliner2_classifier.py --run-name "high_lr_experiment"
  
  # With custom settings
  python train_gliner2_classifier.py --epochs 15 --batch-size 16
  
  # Using LoRA for parameter-efficient training
  python train_gliner2_classifier.py --use-lora --lora-r 16
  
  # List all available checkpoints
  python train_gliner2_classifier.py --list-checkpoints
  
  # Evaluate the latest model
  python train_gliner2_classifier.py --eval-only latest
  
  # Evaluate a specific checkpoint
  python train_gliner2_classifier.py --eval-only run_20260110_143052

Checkpoint Management:
  - Each training run creates a NEW checkpoint (previous models are preserved)
  - Models are stored in: ./gliner2_ad_classifier/runs/run_{timestamp}/
  - A 'latest' symlink always points to the most recent successful model
  - Use --list-checkpoints to see all available models
  - Use --eval-only {run_id} to evaluate a specific checkpoint
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
                        help='Base directory for all model checkpoints')
    
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
    
    # Run identification
    parser.add_argument('--run-name', type=str, default=None,
                        help='Optional name suffix for this training run (e.g., "high_lr_experiment")')
    parser.add_argument('--notes', type=str, default=None,
                        help='Optional notes about this training run')
    
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
                        help='Evaluate model: "latest", a run_id, or full path')
    parser.add_argument('--list-checkpoints', action='store_true',
                        help='List all available model checkpoints and exit')
    
    args = parser.parse_args()
    
    # List checkpoints mode
    if args.list_checkpoints:
        print_checkpoints(args.output_dir)
        return
    
    # Evaluation mode
    if args.eval_only:
        # Resolve the model path
        if args.eval_only == "latest":
            model_path = get_latest_model_path(args.output_dir)
            if not model_path:
                print(f"❌ No trained models found in: {args.output_dir}")
                return
            print(f"📍 Using latest model: {model_path}")
        elif args.eval_only.startswith("run_"):
            # It's a run ID
            model_path = get_model_by_run_id(args.output_dir, args.eval_only)
            if not model_path:
                print(f"❌ Model not found for run: {args.eval_only}")
                print("   Use --list-checkpoints to see available models")
                return
            print(f"📍 Using model from run: {args.eval_only}")
        else:
            # It's a direct path
            model_path = args.eval_only
        
        evaluate_model(model_path, args.val_data)
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
    
    # Show existing checkpoints before training
    existing_checkpoints = list_available_checkpoints(args.output_dir)
    if existing_checkpoints:
        print(f"\n📋 Found {len(existing_checkpoints)} existing checkpoint(s)")
        print("   New training will create a NEW checkpoint (previous models preserved)")
    
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
            run_name=args.run_name,
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
            run_name=args.run_name,
            notes=args.notes,
        )
    
    # Evaluate the trained model
    print("\n" + "=" * 70)
    print("📊 Final Evaluation on Validation Set")
    print("=" * 70)
    evaluate_model(model_path, args.val_data)


if __name__ == "__main__":
    main()
