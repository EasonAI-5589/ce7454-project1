#!/bin/bash

# Quick Start Script for Face Parsing Training
# Usage: ./quick_start.sh [train|resume|test]

set -e  # Exit on error

echo "=========================================="
echo "Face Parsing Training - MicroSegFormer"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if data exists
check_data() {
    if [ ! -d "data/train/images" ]; then
        echo -e "${RED}❌ Error: Training data not found!${NC}"
        echo "Please ensure data is in: data/train/images/"
        exit 1
    fi

    train_count=$(ls -1 data/train/images/*.jpg 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Found $train_count training images${NC}"
}

# Train from scratch
train() {
    echo -e "\n${YELLOW}Starting training...${NC}"
    python main.py --config configs/main.yaml
}

# Resume from checkpoint
resume() {
    echo -e "\n${YELLOW}Resuming training...${NC}"
    if [ -z "$1" ]; then
        echo -e "${RED}Error: Please specify checkpoint path${NC}"
        echo "Usage: ./quick_start.sh resume <checkpoint_path>"
        exit 1
    fi
    python main.py --config configs/main.yaml --resume "$1"
}

# Test model
test() {
    echo -e "\n${YELLOW}Testing model...${NC}"
    if [ -z "$1" ]; then
        echo -e "${RED}Error: Please specify checkpoint path${NC}"
        echo "Usage: ./quick_start.sh test <checkpoint_path>"
        exit 1
    fi
    python test.py --checkpoint "$1"
}

# Show usage
usage() {
    echo ""
    echo "Usage: ./quick_start.sh [command] [args]"
    echo ""
    echo "Commands:"
    echo "  train                    - Start training from scratch"
    echo "  resume <checkpoint>      - Resume training from checkpoint"
    echo "  test <checkpoint>        - Test model on validation set"
    echo ""
    echo "Examples:"
    echo "  ./quick_start.sh train"
    echo "  ./quick_start.sh resume checkpoints/best_model.pth"
    echo "  ./quick_start.sh test checkpoints/best_model.pth"
    echo ""
}

# Main
main() {
    check_data

    if [ $# -eq 0 ]; then
        # Default: start training
        train
    else
        case $1 in
            train)
                train
                ;;
            resume)
                resume "$2"
                ;;
            test)
                test "$2"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown command: $1${NC}"
                usage
                exit 1
                ;;
        esac
    fi
}

main "$@"
