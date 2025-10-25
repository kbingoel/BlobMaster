import subprocess
import sys

failures = 0
runs = 20

for i in range(runs):
    result = subprocess.run(
        ['venv/Scripts/python.exe', '-m', 'pytest',
         'ml/mcts/test_mcts.py::TestMCTSSearch::test_search_respects_dealer_constraint',
         '-q'],
        capture_output=True
    )
    if result.returncode != 0:
        failures += 1
        print(f"Run {i+1}: FAILED")
    else:
        print(f"Run {i+1}: PASSED")

print(f"\n{runs-failures}/{runs} passed, {failures}/{runs} failed")
sys.exit(failures > 0)
