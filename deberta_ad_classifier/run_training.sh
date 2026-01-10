#!/bin/bash
# ==============================================================================
# DeBERTa Ad Classifier Training Script
# Fine-tunes microsoft/deberta-v3-base for podcast ad detection
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  DeBERTa Ad Classifier Training${NC}"
echo -e "${GREEN}======================================${NC}"

# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "\n${YELLOW}Project root:${NC} $PROJECT_ROOT"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
else
    echo -e "${YELLOW}No virtual environment found. Using system Python.${NC}"
fi

# Check Python version
python_version=$(python3 --version 2>&1)
echo -e "${YELLOW}Python version:${NC} $python_version"

# Check CUDA availability
echo -e "\n${YELLOW}Checking GPU availability...${NC}"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null || echo "Could not check CUDA - torch may not be installed"

# Check if training data exists
if [ ! -f "training_data/train_latest.jsonl" ]; then
    echo -e "${RED}Error: Training data not found at training_data/train_latest.jsonl${NC}"
    exit 1
fi

if [ ! -f "training_data/val_latest.jsonl" ]; then
    echo -e "${RED}Error: Validation data not found at training_data/val_latest.jsonl${NC}"
    exit 1
fi

train_count=$(wc -l < "training_data/train_latest.jsonl")
val_count=$(wc -l < "training_data/val_latest.jsonl")
echo -e "${GREEN}Training samples:${NC} $train_count"
echo -e "${GREEN}Validation samples:${NC} $val_count"

# Install/upgrade dependencies if needed
echo -e "\n${YELLOW}Checking dependencies...${NC}"
pip install -q transformers datasets accelerate scikit-learn torch --upgrade 2>/dev/null || {
    echo -e "${RED}Failed to install dependencies. Please run:${NC}"
    echo "pip install -r deberta_ad_classifier/requirements.txt"
}

# Run training
echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}  Starting Training...${NC}"
echo -e "${GREEN}======================================${NC}\n"

# Default training with config file
python3 deberta_ad_classifier/train_deberta.py --config deberta_ad_classifier/config.json "$@"

echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}  Training Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
