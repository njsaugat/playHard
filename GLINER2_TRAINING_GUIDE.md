# GLiNER2 Ad Detection: Complete Training, Testing & Production Guide

> A Senior ML Engineer's guide to building a production-ready ad detection system

## Table of Contents
1. [Overview](#overview)
2. [Understanding Your Current Architecture](#understanding-your-current-architecture)
3. [Data Preparation Strategy](#data-preparation-strategy)
4. [Model Training Workflow](#model-training-workflow)
5. [Testing & Evaluation Framework](#testing--evaluation-framework)
6. [Production Deployment Checklist](#production-deployment-checklist)
7. [Optimization Strategies](#optimization-strategies)

---

## 1. Overview

You have two core tasks:
1. **Binary Classification**: Is this segment an Ad or Not an Ad?
2. **Entity Extraction**: Extract sponsor name, product, promo code, URL, etc.

GLiNER2 can handle both in a single model using its unified architecture.

---

## 2. Understanding Your Current Architecture

### Data Flow
```
VerifiedEpisodeSponsor (Human-verified labels)
         │
         ▼
┌─────────────────────────────────────┐
│  prepare_training_data.py           │
│  - Fetches from DB                  │
│  - Extracts transcript blurbs       │
│  - Creates JSONL for GLiNER2        │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  train_gliner2_classifier.py        │
│  - Loads base GLiNER2 model         │
│  - Fine-tunes on your data          │
│  - Saves best checkpoint            │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  ad_detector.py                     │
│  - Uses trained model               │
│  - Multi-signal scoring             │
│  - Exclusion filters                │
└─────────────────────────────────────┘
```

### Key Insight: Your VerifiedEpisodeSponsor Table

```prisma
model VerifiedEpisodeSponsor {
  is_deleted        Boolean   # TRUE = NOT an ad, FALSE = IS an ad
  deletion_reason   String?   # Why it's not an ad (brand_love, patreon, etc.)
  is_manually_verified Boolean  # Human verified = high quality label
  manually_updated_details Json? # Corrected timing/entities
}
```

This is **gold-standard training data** because:
- Human-verified labels (is_manually_verified = true)
- Negative examples with reasons (is_deleted = true + deletion_reason)
- Corrected boundaries (manually_updated_details.start_time/end_time)

---

## 3. Data Preparation Strategy

### Step 1: Extract Training Data from Database

```bash
# Basic extraction (all verified data)
python prepare_training_data.py

# With limit for testing
python prepare_training_data.py --limit 1000

# Include deletion reasons for multi-task learning
python prepare_training_data.py --include-reason
```

### Step 2: Data Quality Checks

Before training, verify your data:

```python
# Inspect training data distribution
import json

with open('./training_data/train_latest.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Count labels
ads = sum(1 for d in data if 'ad' in d['output']['classifications'][0]['true_label'])
non_ads = len(data) - ads

print(f"Total: {len(data)}")
print(f"Ads: {ads} ({ads/len(data)*100:.1f}%)")
print(f"Non-Ads: {non_ads} ({non_ads/len(data)*100:.1f}%)")

# Check class balance - should be roughly 50/50 or use weighted sampling
if ads / len(data) < 0.3 or ads / len(data) > 0.7:
    print("⚠️ CLASS IMBALANCE DETECTED - Consider oversampling minority class")
```

### Step 3: Creating Entity Extraction Training Data

For entity extraction (sponsor names, products, etc.), you need a different format:

```python
# Create entity extraction training data
def create_entity_training_data(verified_sponsors):
    """
    Create JSONL for entity extraction training.
    
    Format required by GLiNER2:
    {
        "input": "This episode is brought to you by BetterHelp...",
        "output": {
            "entities": [
                {"text": "BetterHelp", "label": "sponsor", "start": 35, "end": 45},
                {"text": "betterhelp.com/podcast", "label": "website", "start": 100, "end": 122},
                {"text": "PODCAST20", "label": "promo_code", "start": 150, "end": 159}
            ]
        }
    }
    """
    examples = []
    
    for record in verified_sponsors:
        text = record['transcript_blurb']
        
        # Get entities from manually_updated_details or ai_updated_details
        details = record.get('manually_updated_details') or record.get('ai_updated_details') or {}
        
        entities = []
        
        # Extract sponsor name
        if details.get('sponsorName'):
            pos = text.lower().find(details['sponsorName'].lower())
            if pos >= 0:
                entities.append({
                    "text": details['sponsorName'],
                    "label": "sponsor",
                    "start": pos,
                    "end": pos + len(details['sponsorName'])
                })
        
        # Extract URL
        if details.get('sponsorUrl'):
            url = details['sponsorUrl']
            pos = text.lower().find(url.lower())
            if pos >= 0:
                entities.append({
                    "text": url,
                    "label": "website",
                    "start": pos,
                    "end": pos + len(url)
                })
        
        # Extract promo code
        if details.get('promoCode'):
            code = details['promoCode']
            pos = text.upper().find(code.upper())
            if pos >= 0:
                entities.append({
                    "text": code,
                    "label": "promo_code",
                    "start": pos,
                    "end": pos + len(code)
                })
        
        if entities:
            examples.append({
                "input": text,
                "output": {"entities": entities}
            })
    
    return examples
```

### Step 4: Combined Multi-Task Training Data

GLiNER2 supports training on multiple tasks simultaneously:

```python
def create_combined_training_data(record):
    """
    Create training example for both classification AND entity extraction.
    """
    return {
        "input": record['transcript_blurb'],
        "output": {
            "classifications": [
                {
                    "task": "ad_detection",
                    "labels": ["ad", "no_ad"],
                    "true_label": [record['label']]
                }
            ],
            "entities": [
                # Add entity annotations here
            ]
        }
    }
```

---

## 4. Model Training Workflow

### Training Options

#### Option 1: Classification Only (Recommended for Start)
```bash
python train_gliner2_classifier.py \
    --train-data ./training_data/train_latest.jsonl \
    --val-data ./training_data/val_latest.jsonl \
    --output-dir ./gliner2_ad_classifier \
    --epochs 10 \
    --batch-size 8
```

#### Option 2: With LoRA (Memory Efficient)
```bash
python train_gliner2_classifier.py \
    --use-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 16
```

#### Option 3: Full Fine-tuning with Custom Hyperparameters
```bash
python train_gliner2_classifier.py \
    --epochs 20 \
    --batch-size 16 \
    --encoder-lr 2e-5 \
    --task-lr 1e-4 \
    --warmup-ratio 0.1 \
    --patience 5 \
    --wandb  # Optional: track experiments
```

### Hyperparameter Tuning Guide

| Parameter | Start Value | Range | Notes |
|-----------|-------------|-------|-------|
| epochs | 10 | 5-30 | Use early stopping |
| batch_size | 8 | 4-32 | Increase if GPU memory allows |
| encoder_lr | 1e-5 | 5e-6 to 5e-5 | Lower = more stable |
| task_lr | 5e-4 | 1e-4 to 1e-3 | Classification head LR |
| warmup_ratio | 0.1 | 0.05-0.2 | % of steps for warmup |
| lora_r | 8 | 4-64 | Higher = more capacity |

---

## 5. Testing & Evaluation Framework

### Level 1: Offline Metrics

Run comprehensive evaluation:

```bash
python train_gliner2_classifier.py --eval-only ./gliner2_ad_classifier/best
```

Key metrics to track:

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| **Precision** | > 0.85 | Minimize false positives (non-ads marked as ads) |
| **Recall** | > 0.90 | Catch most actual ads |
| **F1 Score** | > 0.87 | Balanced performance |
| **AUC-ROC** | > 0.92 | Overall discrimination ability |

### Level 2: Error Analysis

Create an error analysis script:

```python
# error_analysis.py
from collections import defaultdict

def analyze_errors(y_true, y_pred, texts, deletion_reasons=None):
    """Detailed error analysis"""
    
    errors = {
        'false_positives': [],  # Marked as ad but isn't
        'false_negatives': [],  # Missed actual ads
    }
    
    for i, (true, pred, text) in enumerate(zip(y_true, y_pred, texts)):
        if true != pred:
            error_type = 'false_positives' if pred == 'ad' else 'false_negatives'
            errors[error_type].append({
                'text': text[:200],
                'true_label': true,
                'predicted': pred,
                'deletion_reason': deletion_reasons[i] if deletion_reasons else None
            })
    
    # Analyze patterns
    print("\n=== FALSE POSITIVES (Wrongly flagged as ads) ===")
    fp_reasons = defaultdict(int)
    for err in errors['false_positives']:
        reason = err.get('deletion_reason', 'unknown')
        fp_reasons[reason] += 1
        print(f"\n[{reason}]: {err['text'][:100]}...")
    
    print("\nFP Breakdown by Reason:")
    for reason, count in sorted(fp_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    print("\n=== FALSE NEGATIVES (Missed ads) ===")
    for err in errors['false_negatives'][:10]:
        print(f"\n{err['text'][:100]}...")
    
    return errors
```

### Level 3: Stratified Testing

Test on different ad types:

```python
def stratified_evaluation(model, test_data):
    """Evaluate on different ad categories"""
    
    categories = {
        'high_confidence_ads': [],      # Clear sponsor mentions
        'subtle_ads': [],               # No explicit "sponsored by"
        'brand_love': [],               # Genuine recommendations
        'patreon_substack': [],         # Support the show
        'social_media_plugs': [],       # Follow us on...
    }
    
    for record in test_data:
        text = record['input'].lower()
        
        # Categorize based on patterns
        if 'sponsored by' in text or 'brought to you' in text:
            categories['high_confidence_ads'].append(record)
        elif 'patreon' in text or 'substack' in text:
            categories['patreon_substack'].append(record)
        elif 'follow' in text and any(x in text for x in ['twitter', 'instagram', 'youtube']):
            categories['social_media_plugs'].append(record)
        # ... etc
    
    # Evaluate each category separately
    for category, records in categories.items():
        if records:
            accuracy = evaluate_subset(model, records)
            print(f"{category}: {accuracy:.2%} ({len(records)} samples)")
```

### Level 4: A/B Testing in Production

```python
class ABTestingWrapper:
    """Compare old vs new model in production"""
    
    def __init__(self, old_model, new_model, sample_rate=0.1):
        self.old_model = old_model
        self.new_model = new_model
        self.sample_rate = sample_rate
        self.comparison_log = []
    
    def classify(self, text):
        # Always use new model for production
        new_result = self.new_model.classify(text)
        
        # Log comparison for sample_rate of requests
        if random.random() < self.sample_rate:
            old_result = self.old_model.classify(text)
            self.comparison_log.append({
                'text': text[:200],
                'old_prediction': old_result,
                'new_prediction': new_result,
                'timestamp': datetime.now()
            })
        
        return new_result
    
    def get_comparison_stats(self):
        agreements = sum(1 for r in self.comparison_log 
                        if r['old_prediction'] == r['new_prediction'])
        return {
            'total_samples': len(self.comparison_log),
            'agreement_rate': agreements / len(self.comparison_log) if self.comparison_log else 0
        }
```

---

## 6. Production Deployment Checklist

### Pre-Deployment Validation

```bash
# 1. Run full evaluation suite
python train_gliner2_classifier.py --eval-only ./gliner2_ad_classifier/best

# 2. Test on held-out dataset (never seen during training)
python evaluate_holdout.py --model ./gliner2_ad_classifier/best --data ./holdout_test.jsonl

# 3. Run latency benchmarks
python benchmark_inference.py --model ./gliner2_ad_classifier/best --samples 1000
```

### Model Validation Criteria

✅ **Go Criteria:**
- F1 Score ≥ 0.85
- Latency p99 < 200ms
- Memory usage < 4GB
- No regression on key categories

❌ **No-Go Criteria:**
- Precision < 0.80 (too many false positives)
- Recall < 0.85 (missing too many ads)
- Error rate on brand love > 30%

### Deployment Strategy

```python
# In ad_detector.py - Use trained model
class AdDetector:
    DEFAULT_MODEL_NAME = "./gliner2_ad_classifier/best"  # Your trained model
    FALLBACK_MODEL = "fastino/gliner2-base-v1"  # Fallback to base
    
    def __init__(self, model_name: str = None):
        model_to_load = model_name or self.DEFAULT_MODEL_NAME
        
        if os.path.exists(model_to_load):
            print(f"📦 Loading trained model: {model_to_load}")
            self.model = GLiNER2.from_pretrained(model_to_load)
        else:
            print(f"⚠️ Trained model not found, using base: {self.FALLBACK_MODEL}")
            self.model = GLiNER2.from_pretrained(self.FALLBACK_MODEL)
```

---

## 7. Optimization Strategies

### Data Quality Optimization

1. **Active Learning**: Query model on uncertain predictions, send for human review
2. **Hard Negative Mining**: Focus on false positives in next training round
3. **Data Augmentation**: Paraphrase ads using LLM

### Model Optimization

1. **Ensemble**: Combine GLiNER2 with rule-based detection
2. **Confidence Calibration**: Calibrate output probabilities
3. **Threshold Tuning**: Find optimal threshold per category

### Continuous Improvement Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS IMPROVEMENT LOOP                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Production Logging                                          │
│     └─> Log predictions + confidence + latency                  │
│                                                                 │
│  2. Human Review Queue                                          │
│     └─> Sample low-confidence predictions for review            │
│     └─> Add to VerifiedEpisodeSponsor table                     │
│                                                                 │
│  3. Weekly Retraining                                           │
│     └─> python prepare_training_data.py                         │
│     └─> python train_gliner2_classifier.py                      │
│                                                                 │
│  4. A/B Testing                                                 │
│     └─> Compare new model vs current production                 │
│     └─> Promote if metrics improve                              │
│                                                                 │
│  5. Monitoring Dashboard                                        │
│     └─> Track precision/recall drift over time                  │
│     └─> Alert on significant degradation                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start Commands

```bash
# Step 1: Prepare data
python prepare_training_data.py

# Step 2: Train model
python train_gliner2_classifier.py --epochs 10

# Step 3: Evaluate
python train_gliner2_classifier.py --eval-only ./gliner2_ad_classifier/best

# Step 4: Test interactively
python use_trained_model.py --demo

# Step 5: Use in production
# Update ad_detector.py to use ./gliner2_ad_classifier/best
```

---

## Next Steps

1. **Generate baseline metrics** with current base model
2. **Run prepare_training_data.py** to create dataset
3. **Review data quality** in CSV files
4. **Train initial model** with default settings
5. **Evaluate and iterate** based on error analysis
