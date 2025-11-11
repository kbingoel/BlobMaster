#!/bin/bash
source venv/bin/activate

# Auto-detect most recent profile file
PROFILE_FILE=$(ls -t profile_*.prof 2>/dev/null | head -1)

if [ -z "$PROFILE_FILE" ]; then
    echo "ERROR: No profile files found!"
    echo "Run profiling first: python ml/profile_selfplay.py"
    exit 1
fi

echo "========================================"
echo "Inspecting: $PROFILE_FILE"
echo "========================================"
echo ""

python -m pstats "$PROFILE_FILE"
