# DeBERTa Ad Classifier

Fine-tuning **microsoft/deberta-v3-base** for detecting advertisements in podcast transcripts.

## Overview

This project trains a DeBERTa-v3-base model for binary text classification to detect ads in podcast audio transcripts. The model learns to distinguish between:
- **ad**: Content that is an advertisement
- **no_ad**: Regular podcast content (not an advertisement)

## Model Architecture

- **Base Model**: `microsoft/deberta-v3-base`
- **Task**: Binary Sequence Classification
- **Architecture**: DeBERTa v3 (Decoding-enhanced BERT with disentangled attention)
- **Parameters**: ~86M (base model)

### Why DeBERTa?

DeBERTa-v3-base is one of the best-in-class models for text classification tasks:
1. **Disentangled Attention**: Separates content and position information for better representation
2. **Enhanced Mask Decoder**: Improves pre-training effectiveness
3. **ELECTRA-style Training**: More sample-efficient pre-training
4. **Strong Performance**: Consistently outperforms BERT/RoBERTa on GLUE benchmarks

## Project Structure

```
deberta_ad_classifier/
├── train_deberta.py      # Main training script
├── predict.py            # Inference/prediction script  
├── config.json           # Training configuration
├── requirements.txt      # Python dependencies
├── run_training.sh       # Shell script to run training
├── README.md            # This file
└── output/              # Training outputs (created during training)
    └── run_YYYYMMDD_HHMMSS/
        ├── best_model/           # Best checkpoint
        ├── training_config.json  # Config used for this run
        ├── training_results.json # Metrics and results
        └── logs/                 # Training logs
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r deberta_ad_classifier/requirements.txt
```

### 2. Train the Model

**Using the shell script:**
```bash
chmod +x deberta_ad_classifier/run_training.sh
./deberta_ad_classifier/run_training.sh
```

**Or directly with Python:**
```bash
python deberta_ad_classifier/train_deberta.py --config deberta_ad_classifier/config.json
```

**With custom parameters:**
```bash
python deberta_ad_classifier/train_deberta.py \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-5
```

### 3. Make Predictions

**Single text:**
```bash
python deberta_ad_classifier/predict.py \
    --model_path deberta_ad_classifier/output/run_XXXXX/best_model \
    --text "This podcast is sponsored by Company X. Get 20% off with code SAVE20."
```

**Interactive mode:**
```bash
python deberta_ad_classifier/predict.py \
    --model_path deberta_ad_classifier/output/run_XXXXX/best_model \
    --interactive
```

**Batch processing:**
```bash
python deberta_ad_classifier/predict.py \
    --model_path deberta_ad_classifier/output/run_XXXXX/best_model \
    --input_file texts.txt \
    --output_file predictions.json
```

## Configuration

Edit `config.json` to customize training:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `microsoft/deberta-v3-base` | Base model |
| `max_length` | 512 | Max sequence length |
| `num_train_epochs` | 5 | Training epochs |
| `per_device_train_batch_size` | 8 | Training batch size |
| `learning_rate` | 2e-5 | Learning rate |
| `weight_decay` | 0.01 | Weight decay |
| `warmup_ratio` | 0.1 | LR warmup ratio |
| `early_stopping_patience` | 3 | Early stopping patience |

## Data Format

Training data should be in JSONL format:

```json
{"input": "Text content here...", "output": {"classifications": [{"task": "ad_detection", "labels": ["ad", "no_ad"], "true_label": ["ad"]}]}}
{"input": "Another text...", "output": {"classifications": [{"task": "ad_detection", "labels": ["ad", "no_ad"], "true_label": ["no_ad"]}]}}
```

## Expected Results

With the default configuration on the podcast ad dataset:

| Metric | Expected Range |
|--------|---------------|
| Accuracy | 85-92% |
| F1 Score | 0.85-0.92 |
| Precision | 0.84-0.91 |
| Recall | 0.86-0.93 |

Results may vary based on data quality and distribution.

## GPU Requirements

- **Minimum**: 8GB VRAM (with gradient accumulation)
- **Recommended**: 16GB+ VRAM for larger batch sizes
- **CPU**: Training possible but significantly slower (10-20x)

## Tips for Better Results

1. **Data Quality**: Ensure training data is clean and well-labeled
2. **Class Balance**: Consider oversampling/undersampling if classes are imbalanced
3. **Hyperparameter Tuning**: Try different learning rates (1e-5 to 5e-5)
4. **Longer Training**: Increase epochs if validation metrics are still improving
5. **Larger Batches**: If GPU memory allows, larger batches often help

## Troubleshooting

**Out of Memory (OOM):**
- Reduce `per_device_train_batch_size` to 4 or 2
- Increase `gradient_accumulation_steps` to compensate

**Slow Training:**
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Enable fp16 training (default when CUDA available)

**Poor Performance:**
- Check data quality and label consistency
- Try a smaller learning rate (1e-5)
- Increase training epochs

## License

MIT License - Feel free to use and modify for your projects.
