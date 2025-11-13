#!/bin/bash
# BlobMaster Phase 1 Training Launcher
# Simple script to start/resume Phase 1 training with optimal settings

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   BlobMaster Phase 1 Training Launcher ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Please create it first:${NC}"
    echo "   python3.14 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r ml/requirements.txt"
    exit 1
fi

echo -e "${GREEN}✓${NC} Activating virtual environment..."
source venv/bin/activate

# Check for existing checkpoints
CHECKPOINT_DIR="models/checkpoints"
LATEST_CHECKPOINT=""
LATEST_ITER=-1

if [ -d "$CHECKPOINT_DIR" ]; then
    # Search in both permanent and cache directories
    for dir in "$CHECKPOINT_DIR/permanent" "$CHECKPOINT_DIR/cache"; do
        if [ -d "$dir" ]; then
            # Find all .pth files and extract iteration numbers
            for checkpoint in "$dir"/*.pth; do
                if [ -f "$checkpoint" ]; then
                    # Extract iteration number from filename (iter###)
                    if [[ "$checkpoint" =~ iter([0-9]+) ]]; then
                        iter="${BASH_REMATCH[1]}"
                        # Remove leading zeros for comparison
                        iter=$((10#$iter))
                        if [ "$iter" -gt "$LATEST_ITER" ]; then
                            LATEST_ITER=$iter
                            LATEST_CHECKPOINT="$checkpoint"
                        fi
                    fi
                fi
            done
        fi
    done
fi

# Build command
CMD="python ml/train.py --workers 32 --training-on rounds --enable-curriculum --iterations 500 --device cuda --save-every 5"

if [ -n "$LATEST_CHECKPOINT" ]; then
    echo -e "${GREEN}✓${NC} Found existing checkpoint at iteration $LATEST_ITER"
    echo -e "   ${LATEST_CHECKPOINT}"
    echo -e "${BLUE}→${NC} Resuming training from iteration $((LATEST_ITER + 1))..."
    echo ""
    CMD="$CMD --resume $LATEST_CHECKPOINT"
else
    echo -e "${GREEN}✓${NC} No existing checkpoints found"
    echo -e "${BLUE}→${NC} Starting fresh Phase 1 training..."
    echo ""
fi

# Display training settings
echo -e "${BLUE}Training Configuration:${NC}"
echo "  • Mode: Phase 1 (Independent Rounds)"
echo "  • Workers: 32 (validated max for RTX 4060 8GB)"
echo "  • Curriculum: Enabled (MCTS 1×15 → 5×50 + Units 2k → 10k)"
echo "  • Iterations: 500"
echo "  • Checkpoints: Every 5 iterations (permanent)"
echo "  • Device: CUDA (RTX 4060)"
echo "  • Expected Duration: ~5 days"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop gracefully (will save checkpoint)${NC}"
echo ""
echo "─────────────────────────────────────────"
echo ""

# Run training
$CMD

echo ""
echo -e "${GREEN}✓ Training completed successfully!${NC}"
