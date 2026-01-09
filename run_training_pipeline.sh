#!/bin/bash
# =============================================================================
# GLiNER2 Ad Detection Training Pipeline
# Complete end-to-end training, evaluation, and deployment workflow
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}▶ $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}\n"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR="./training_data"
MODEL_OUTPUT="./gliner2_ad_classifier"
EPOCHS=10
BATCH_SIZE=8
ENCODER_LR="1e-5"
TASK_LR="5e-4"
USE_LORA=false
LIMIT=""  # Set to number to limit data

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --use-lora)
            USE_LORA=true
            shift
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --output-dir)
            MODEL_OUTPUT="$2"
            shift 2
            ;;
        --skip-data-prep)
            SKIP_DATA_PREP=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --epochs N         Number of training epochs (default: 10)"
            echo "  --batch-size N     Batch size (default: 8)"
            echo "  --use-lora         Use LoRA for memory-efficient training"
            echo "  --limit N          Limit number of training examples"
            echo "  --output-dir DIR   Model output directory"
            echo "  --skip-data-prep   Skip data preparation step"
            echo "  --skip-training    Skip training, only evaluate"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Step 0: Environment Check
# =============================================================================
print_step "Step 0: Environment Check"

# Check Python
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
print_success "Python: $PYTHON_VERSION"

# Check required packages
python -c "import gliner2" 2>/dev/null || {
    print_warning "GLiNER2 not installed. Installing..."
    pip install gliner2
}
print_success "GLiNER2 installed"

python -c "import psycopg2" 2>/dev/null || {
    print_warning "psycopg2 not installed. Installing..."
    pip install psycopg2-binary
}
print_success "psycopg2 installed"

# Check database connection
if [ -z "$DATABASE_URL" ] && [ ! -f ".env.local" ]; then
    print_error "DATABASE_URL not set and .env.local not found"
    exit 1
fi
print_success "Environment configured"

# =============================================================================
# Step 1: Prepare Training Data
# =============================================================================
if [ "$SKIP_DATA_PREP" != "true" ]; then
    print_step "Step 1: Preparing Training Data"
    
    echo "Fetching verified sponsor data from database..."
    python prepare_training_data.py --output-dir "$OUTPUT_DIR" $LIMIT
    
    if [ $? -eq 0 ]; then
        print_success "Training data prepared in $OUTPUT_DIR"
        
        # Show stats
        if [ -f "$OUTPUT_DIR/dataset_stats.json" ]; then
            echo ""
            echo "Dataset Statistics:"
            cat "$OUTPUT_DIR/dataset_stats.json" | python -m json.tool
        fi
    else
        print_error "Failed to prepare training data"
        exit 1
    fi
else
    print_warning "Skipping data preparation (--skip-data-prep)"
fi

# =============================================================================
# Step 2: Train the Model
# =============================================================================
if [ "$SKIP_TRAINING" != "true" ]; then
    print_step "Step 2: Training GLiNER2 Model"
    
    TRAIN_CMD="python train_gliner2_classifier.py"
    TRAIN_CMD="$TRAIN_CMD --train-data $OUTPUT_DIR/train_latest.jsonl"
    TRAIN_CMD="$TRAIN_CMD --val-data $OUTPUT_DIR/val_latest.jsonl"
    TRAIN_CMD="$TRAIN_CMD --output-dir $MODEL_OUTPUT"
    TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
    TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
    TRAIN_CMD="$TRAIN_CMD --encoder-lr $ENCODER_LR"
    TRAIN_CMD="$TRAIN_CMD --task-lr $TASK_LR"
    
    if [ "$USE_LORA" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --use-lora"
    fi
    
    echo "Running: $TRAIN_CMD"
    echo ""
    
    $TRAIN_CMD
    
    if [ $? -eq 0 ]; then
        print_success "Model trained successfully"
        print_success "Model saved to: $MODEL_OUTPUT/best"
    else
        print_error "Training failed"
        exit 1
    fi
else
    print_warning "Skipping training (--skip-training)"
fi

# =============================================================================
# Step 3: Evaluate the Model
# =============================================================================
print_step "Step 3: Evaluating Model"

BEST_MODEL="$MODEL_OUTPUT/best"

if [ ! -d "$BEST_MODEL" ]; then
    print_warning "Best model not found at $BEST_MODEL, using model directory"
    BEST_MODEL="$MODEL_OUTPUT"
fi

echo "Evaluating model: $BEST_MODEL"
echo ""

python evaluate_model.py \
    --model "$BEST_MODEL" \
    --test-data "$OUTPUT_DIR/val_latest.jsonl" \
    --analyze-errors \
    --output "./evaluation_results.json"

if [ $? -eq 0 ]; then
    print_success "Evaluation complete"
    print_success "Results saved to: ./evaluation_results.json"
else
    print_warning "Evaluation completed with warnings"
fi

# =============================================================================
# Step 4: Compare with Baseline
# =============================================================================
print_step "Step 4: Comparing with Baseline Model"

python evaluate_model.py \
    --model "$BEST_MODEL" \
    --test-data "$OUTPUT_DIR/val_latest.jsonl" \
    --compare-baseline

# =============================================================================
# Step 5: Find Optimal Threshold
# =============================================================================
print_step "Step 5: Finding Optimal Threshold"

python evaluate_model.py \
    --model "$BEST_MODEL" \
    --test-data "$OUTPUT_DIR/val_latest.jsonl" \
    --find-threshold

# =============================================================================
# Step 6: Demo the Model
# =============================================================================
print_step "Step 6: Running Demo"

python use_trained_model.py --model-path "$BEST_MODEL" --demo

# =============================================================================
# Summary
# =============================================================================
print_step "Pipeline Complete! 🎉"

echo ""
echo "Summary:"
echo "  - Training data: $OUTPUT_DIR/"
echo "  - Trained model: $MODEL_OUTPUT/best"
echo "  - Evaluation results: ./evaluation_results.json"
echo ""
echo "Next steps:"
echo "  1. Review evaluation results"
echo "  2. If metrics are satisfactory, update ad_detector.py to use trained model"
echo "  3. Test in production with A/B testing"
echo ""
echo "To use in ad_detector.py:"
echo "  detector = AdDetector(model_name='$BEST_MODEL')"
echo ""

print_success "Done!"
