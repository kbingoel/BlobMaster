#!/usr/bin/env python3
"""Verify that all fixed tests pass consistently."""

import subprocess
import sys

# Tests to verify
tests = [
    "ml/tests/test_integration.py::TestMCTSIntegration::test_integration_bidding",
    "ml/tests/test_integration.py::TestQualityValidation::test_mcts_vs_random_baseline",
    "ml/tests/test_integration.py::TestQualityValidation::test_all_moves_legal",
    "ml/mcts/test_mcts.py::TestMCTSIntegrationComplete::test_batched_inference_complete_bidding",
]

print("=" * 80)
print("VERIFYING FIXED TESTS")
print("=" * 80)

failed_tests = []

for test in tests:
    print(f"\n{'=' * 80}")
    print(f"Running: {test}")
    print("=" * 80)

    # Run each test 5 times to check for flakiness
    passes = 0
    failures = 0

    for run in range(5):
        result = subprocess.run(
            ["venv/Scripts/python.exe", "-m", "pytest", test, "-v", "-x"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            passes += 1
            print(f"  Run {run + 1}: PASS")
        else:
            failures += 1
            print(f"  Run {run + 1}: FAIL")
            print(f"    Output: {result.stdout[-200:]}")

    print(f"\n{test}: {passes}/5 passed")

    if failures > 0:
        failed_tests.append((test, passes, failures))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if not failed_tests:
    print("✅ All tests passed consistently (5/5 runs each)")
    sys.exit(0)
else:
    print(f"❌ {len(failed_tests)} test(s) had failures:")
    for test, passes, failures in failed_tests:
        print(f"  - {test}: {passes}/5 passed ({failures} failures)")
    sys.exit(1)
