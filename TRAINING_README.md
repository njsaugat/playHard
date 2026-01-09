# GLiNER2 Ad Classifier Training Pipeline

Train a custom GLiNER2 model for podcast ad detection using your verified sponsor data.

**Reference**: [GLiNER2 Training Documentation](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/9-training.md)

## Overview

This pipeline:
1. **Extracts** verified sponsor data from your database
2. **Processes** transcripts to extract ad/non-ad segments  
3. **Formats** data for GLiNER2 training (JSONL)
4. **Trains** a classification model using the official GLiNER2 API
5. **Evaluates** performance on held-out data

## Quick Start

### 1. Prepare Training Data

```bash
# Fetch all verified sponsors and create training dataset
python prepare_training_data.py

# Or limit to N records for testing
python prepare_training_data.py --limit 1000

# Include deletion reason as multi-task learning
python prepare_training_data.py --include-reason
```

This creates:
- `training_data/train_latest.jsonl` - Training data
- `training_data/val_latest.jsonl` - Validation data  
- `training_data/*.csv` - CSV files for inspection
- `training_data/dataset_stats.json` - Dataset statistics

### 2. Train the Model

```bash
# Basic training (uses official GLiNER2 API)
python train_gliner2_classifier.py

# Custom configuration
python train_gliner2_classifier.py \
    --epochs 15 \
    --batch-size 16 \
    --encoder-lr 2e-5 \
    --task-lr 1e-4 \
    --output-dir ./my_ad_classifier

# Using LoRA for parameter-efficient training (less VRAM)
python train_gliner2_classifier.py --use-lora --lora-r 16

# Train directly from JSONL files (simpler)
python train_gliner2_classifier.py --jsonl-mode

# Evaluate an existing model
python train_gliner2_classifier.py --eval-only ./gliner2_ad_classifier/best
```

### 3. Use the Trained Model

```python
from use_trained_model import TrainedAdClassifier

# Load your trained model
classifier = TrainedAdClassifier("./gliner2_ad_classifier/best")

# Classify text
is_ad, confidence, label = classifier.classify(transcript_text)

if is_ad:
    print(f"AD detected with {confidence:.0%} confidence")
```

## Data Format

### Input from Database

```sql
SELECT 
    PodcastEpi.id as "episodeId", 
    PodcastEpi."transcriptUrl",
    EpiSponsor.start_time,
    EpiSponsor.end_time,
    VerifiedSpo.ai_updated_details,
    VerifiedSpo.is_deleted,            -- True = NOT ad, False = IS ad
    VerifiedSpo.deletion_reason,
    VerifiedSpo.deletion_description, 
    VerifiedSpo.is_manually_verified,
    VerifiedSpo.manually_updated_details 
FROM ...
WHERE VerifiedSpo.is_manually_verified = true
```

### Timing Priority

1. `manually_updated_details` → Highest priority
2. `ai_updated_details` → Second priority  
3. DB columns (`start_time`, `end_time`) → Fallback

### GLiNER2 JSONL Format

```json
{
  "input": "This episode is brought to you by BetterHelp...",
  "output": {
    "classifications": [
      {
        "task": "ad_detection",
        "labels": ["ad", "no_ad"],
        "true_label": ["ad"]
      }
    ]
  }
}
```

## Training Configuration

Uses official GLiNER2 `TrainingConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 8 | Batch size |
| `--encoder-lr` | 1e-5 | Learning rate for encoder (pretrained) |
| `--task-lr` | 5e-4 | Learning rate for classification head |
| `--warmup-ratio` | 0.1 | Warmup ratio |
| `--use-lora` | False | Use LoRA for efficient training |
| `--lora-r` | 8 | LoRA rank |
| `--patience` | 3 | Early stopping patience |

### Hardware Requirements

- **GPU**: Recommended for training (8GB+ VRAM)
- **CPU**: Possible but slower
- **RAM**: 16GB+ recommended

## Files

| File | Description |
|------|-------------|
| `prepare_training_data.py` | Fetches and processes data from DB |
| `train_gliner2_classifier.py` | Trains the GLiNER2 model (official API) |
| `use_trained_model.py` | Utility to use trained model |
| `ad_detector.py` | Original ad detector (uses base GLiNER2) |

## Integrating with Existing Pipeline

### Option 1: Modify `ad_detector.py`

```python
class AdDetector:
    def __init__(self, model_name: str = None):
        if model_name and os.path.exists(model_name):
            self.model = GLiNER2.from_pretrained(model_name)
        else:
            self.model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Usage:
detector = AdDetector(model_name="./gliner2_ad_classifier/best")
```

### Option 2: Hybrid Approach

Use trained classifier for classification, base model for entity extraction:

```python
from ad_detector import AdDetector
from use_trained_model import TrainedAdClassifier

base_detector = AdDetector()
classifier = TrainedAdClassifier("./gliner2_ad_classifier/best")

# Get entities from base detector
segment = base_detector.analyze_segment(text)

# Override classification with trained model
is_ad, confidence, _ = classifier.classify(text)
segment.is_ad_classification = is_ad
segment.classification_confidence = confidence
```

## Troubleshooting

### Out of Memory

```bash
# Use LoRA for memory-efficient training
python train_gliner2_classifier.py --use-lora --batch-size 4
```

### Slow Training on CPU

```bash
# Use smaller dataset for testing
python prepare_training_data.py --limit 500
python train_gliner2_classifier.py --epochs 5
```

### No CUDA Available

Training will automatically use CPU. For faster training:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Performance Tips

1. **Start small**: Test with `--limit 1000` first
2. **Use early stopping**: Enabled by default, stops when val loss stops improving
3. **Try LoRA**: `--use-lora` uses less memory and can be faster
4. **Balance dataset**: Ensure similar ad/non-ad counts
5. **Review CSV**: Check extracted data quality before training
6. **Monitor with W&B**: Use `--wandb` for experiment tracking
