# How to Train BlobMaster

Quick guide for running multi-day training with external monitoring.

## Quick Start

### 1. Start Training

```bash
# Full training run (~5 days with curriculum enabled)
python ml/train.py --training-on rounds --iterations 500 --enable-curriculum

# Fast test (5 minutes)
python ml/train.py --fast --iterations 10
```

### 2. Launch Monitor (separate terminal)

```bash
python ml/monitor.py
```

**Controls:**
- `p` - Pause training (stops after current iteration)
- `q` - Quit monitor (training continues)

### 3. Recommended: Use tmux

```bash
# Terminal 1: Start training in tmux
tmux new -s training
python ml/train.py --training-on rounds --iterations 500
# Detach: Ctrl+B, then D

# Terminal 2: Monitor (can attach/detach anytime)
python ml/monitor.py

# Reattach to training:
tmux attach -t training
```

---

## Common Operations

### Pause Training

**From monitor:** Press `p`

**Manually:**
```bash
echo "PAUSE" > models/checkpoints/control.signal
```

### Resume Training

```bash
# Find latest checkpoint
ls -lt models/checkpoints/cache/

# Resume
python ml/train.py --resume models/checkpoints/cache/[checkpoint].pth --iterations 500
```

### Check Progress

```bash
# View status file
cat models/checkpoints/status.json | jq .

# Or just run the monitor
python ml/monitor.py
```

### Manage Disk Space

```bash
# Check sizes
du -sh models/checkpoints/*/

# Remove old cache (automatic rotation keeps only 4 most recent)
rm models/checkpoints/cache/*.pth

# Remove old permanent checkpoints (manual review recommended)
ls -lt models/checkpoints/permanent/
rm models/checkpoints/permanent/[old-checkpoint].pth
```

---

## Checkpoint Strategy

**Automatic rotation:**
- **Permanent:** Every 5 iterations (5, 10, 15, ..., 500) → `models/checkpoints/permanent/`
- **Cache:** All others, keeps 4 most recent → `models/checkpoints/cache/`

**Disk usage:**
- Without rotation: ~250GB (500 iterations)
- With rotation: ~50GB (100 permanent + 4 cache)

---

## Training Configurations

### Phase 1: Independent Rounds (Current)
```bash
python ml/train.py --training-on rounds --iterations 500 --enable-curriculum

# Timeline: ~5 days with curriculum (MCTS 1×15 → 5×50 + units 2K → 10K)
# Without curriculum: ~72-136 days (depending on fixed MCTS config)
# Performance: Variable with curriculum, ~310-1200 rounds/min depending on stage
```

### Phase 2: Full Games (Future)
```bash
python ml/train.py --training-on games --iterations 100

# Timeline: ~35-40 days (Medium MCTS)
# Performance: ~73 games/min
```

---

## Monitor Details

**Status displayed:**
- Iteration progress (150/500, 30%)
- Phase & MCTS config (rounds, 3×35)
- ETA (days and hours)
- ELO rating with changes
- Learning rate
- Loss metrics (total, policy, value)
- Training units/iteration
- Target temperature (τ)

**Refresh rate:**
```bash
# Default 5 seconds
python ml/monitor.py

# Custom interval
python ml/monitor.py --refresh-interval 10
```

**Platform:** Unix/Linux only (Windows users: use WSL)

---

## Troubleshooting

**Training not visible in monitor:**
```bash
# Check training is running
ps aux | grep train.py

# Check status file exists
ls -l models/checkpoints/status.json
```

**Pause not working:**
```bash
# Verify signal file created
ls -l models/checkpoints/control.signal

# Pause happens at iteration boundary (may take minutes)
```

**Out of disk space:**
```bash
# Check available space
df -h models/checkpoints/

# Emergency cleanup
rm models/checkpoints/cache/*.pth
```

**Monitor keyboard not responding:**
```bash
# Reset terminal
stty sane

# Restart monitor
python ml/monitor.py
```

---

## Files Created During Training

```
models/checkpoints/
├── permanent/                     # Every 5th iteration (kept forever)
│   └── 20251113-Blobmaster-v1-4.9M-iter005-elo1234.pth
├── cache/                         # Others (max 4, auto-rotated)
│   └── 20251113-Blobmaster-v1-4.9M-iter007.pth
├── status.json                    # Live progress (atomic updates)
├── control.signal                 # Pause command (created by monitor)
├── best_[checkpoint].pth          # Best model (updated on promotion)
└── metrics_history.json           # Full training metrics log
```

---

## Expected Training Progression

**With curriculum** (~5 days total):
- **Day 1** (ELO ~800): Random legal moves, learning basic rules
- **Day 3** (ELO ~1200): Consistent trick-taking, early bidding strategy
- **Day 5** (ELO ~1400-1600): Strategic bidding, card counting, suit tracking

**Without curriculum** (fixed config, for reference):
- ~72-136 days depending on MCTS configuration

---

## Next Steps After Training

1. **Evaluate model:**
   ```bash
   # Test on full games
   python ml/evaluate.py --checkpoint models/checkpoints/best_*.pth
   ```

2. **Export to ONNX:** (Phase 5 - not yet implemented)
   ```bash
   python ml/export_onnx.py --checkpoint models/checkpoints/best_*.pth
   ```

3. **Deploy to production:** (Phases 6-7 - not yet implemented)
   - Backend API with ONNX inference
   - Svelte frontend UI

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python ml/train.py --iterations 500` | Start training |
| `python ml/monitor.py` | Launch monitor |
| `echo "PAUSE" > models/checkpoints/control.signal` | Pause training |
| `python ml/train.py --resume [checkpoint]` | Resume training |
| `cat models/checkpoints/status.json \| jq .` | View status |
| `ls -lt models/checkpoints/cache/` | Find latest checkpoint |
| `tmux attach -t training` | Reattach to training session |

---

**Need help?** See [CLAUDE.md](CLAUDE.md) for full project documentation.
