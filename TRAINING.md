# BlobMaster Training Guide

Simple guide for running Phase 1 training on Ubuntu Linux with RTX 4060.

## Quick Start

```bash
# One-time setup (if not done already)
python3.14 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r ml/requirements.txt
deactivate

# Start/resume training (handles everything automatically)
./start_training.sh
```

That's it! The script automatically:
- Activates the virtual environment
- Detects and resumes from the latest checkpoint
- Uses optimal Phase 1 settings (32 workers, curriculum enabled)
- Runs for 500 iterations

## Stopping Training

**Graceful shutdown** (saves checkpoint before exit):
```bash
Press Ctrl+C in the terminal
```

The current iteration will complete, then a checkpoint will be saved. You can resume later with the same `./start_training.sh` command.

## Monitoring Progress

**Real-time console output** shows:
- Current iteration (X/500)
- Games/rounds per minute
- ELO rating progression
- Training loss
- Estimated time remaining

**Checkpoint files** are saved to:
- `models/checkpoints/permanent/` - Every 5th iteration (kept forever)
- `models/checkpoints/cache/` - All others (last 4 kept)

**Status file** (external monitoring):
```bash
cat models/checkpoints/status.json
```

## Training Timeline

**Phase 1** (Independent Rounds, 500 iterations):
- **With curriculum** (recommended): **~5 days** total
  - MCTS curriculum: 1×15 → 5×50 progressive increase
  - Training units: 2K → 10K linear ramp
  - Saves ~3-4 days vs fixed configuration

- **Without curriculum** (fixed config, for reference):
  - Medium MCTS (3×30): ~136 days @ 36.7 games/min
  - Light MCTS (2×20): ~72 days @ 69.1 games/min

The `start_training.sh` script automatically enables the adaptive curriculum for optimal training efficiency.

## What Happens During Training

**Iteration loop** (repeats 500 times):
1. **Self-play**: 32 parallel workers generate games using current model + MCTS
2. **Replay buffer**: Store game positions and MCTS policies (max 500k positions)
3. **Training**: Update neural network on batches from replay buffer
4. **Evaluation**: Test new model vs current best (400 games)
5. **Promotion**: If new model wins >55%, it becomes the new best
6. **Checkpoint**: Save model state, ELO rating, and training metrics

**Adaptive curriculum** (automatic):
- **MCTS search**: 1×15 sims (iter 0-50) → 5×50 sims (iter 450-500)
- **Training units**: 2,000 rounds (iter 0) → 10,000 rounds (iter 500)

This saves ~3-4 days of training time vs fixed settings.

## Advanced Usage

If you need to customize settings, use `ml/train.py` directly:

```bash
source venv/bin/activate

# Custom worker count
python ml/train.py --workers 16 --training-on rounds --enable-curriculum --iterations 500

# Resume from specific checkpoint
python ml/train.py --resume models/checkpoints/permanent/20251113-Blobmaster-v1-32w-rounds-3x35-iter150-elo1420.pth --iterations 500

# Fast test run (validates pipeline)
python ml/train.py --fast --iterations 5
```

## Troubleshooting

**CUDA out of memory**: Reduce `--workers` (default is 32, validated max for RTX 4060 8GB)

**Module not found**: Ensure venv is activated and requirements installed

**Permission denied** on `start_training.sh`: Run `chmod +x start_training.sh`

**Training crashes mid-iteration**: Resume with `./start_training.sh` - it will continue from the last checkpoint

## Next Steps

After Phase 1 completes (500 iterations):
- Phase 2 will train on full multi-round games (not yet implemented)
- Phase 5 will export to ONNX for production inference
- Phases 6-7 will add web UI and deployment
