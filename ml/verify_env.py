"""
Verify Phase 2 development environment is ready.

This script checks that all necessary dependencies for Phase 2 (MCTS + Neural Network)
are properly installed and configured.
"""
import sys


def verify_environment():
    """Verify all Phase 2 dependencies are installed and working."""
    print("=" * 70)
    print("Phase 2 Environment Verification")
    print("=" * 70)

    all_checks_passed = True

    # Check Python version
    print(f"\n✓ Python version: {sys.version.split()[0]}")
    print(f"  Full version: {sys.version}")

    # Check PyTorch
    print("\n" + "-" * 70)
    print("PyTorch (Core ML Framework)")
    print("-" * 70)
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU device: {torch.cuda.get_device_name(0)}")
            print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("  ⚠️  CUDA not available - will use CPU for training (slower)")
            print("  ⚠️  This is expected if you don't have an NVIDIA GPU")
    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        all_checks_passed = False

    # Check NumPy
    print("\n" + "-" * 70)
    print("NumPy (Array Operations)")
    print("-" * 70)
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.version.version}")
    except ImportError as e:
        print(f"✗ NumPy not installed: {e}")
        all_checks_passed = False

    # Check pytest
    print("\n" + "-" * 70)
    print("Testing Framework")
    print("-" * 70)
    try:
        import pytest
        print(f"✓ pytest version: {pytest.__version__}")
    except ImportError as e:
        print(f"✗ pytest not installed: {e}")
        all_checks_passed = False

    # Check ONNX
    print("\n" + "-" * 70)
    print("ONNX (Model Export)")
    print("-" * 70)
    try:
        import onnx
        print(f"✓ ONNX version: {onnx.__version__}")
    except ImportError as e:
        print(f"⚠️  ONNX not installed: {e}")
        print("  Note: ONNX needed for Phase 5 (model export), not Phase 2")

    try:
        import onnxruntime
        print(f"✓ ONNX Runtime version: {onnxruntime.__version__}")
    except ImportError as e:
        print(f"⚠️  ONNX Runtime not installed: {e}")
        print("  Note: ONNX Runtime needed for Phase 5 (inference), not Phase 2")

    # Check optional but useful packages
    print("\n" + "-" * 70)
    print("Optional Packages")
    print("-" * 70)

    try:
        import tensorboard
        print(f"✓ TensorBoard version: {tensorboard.__version__}")
    except ImportError:
        print("⚠️  TensorBoard not installed (useful for monitoring training)")

    try:
        import tqdm
        print(f"✓ tqdm version: {tqdm.__version__}")
    except ImportError:
        print("⚠️  tqdm not installed (useful for progress bars)")

    # Summary
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✅ Environment ready for Phase 2 implementation!")
        print("\nCore dependencies verified:")
        print("  - PyTorch (neural network framework)")
        print("  - NumPy (array operations)")
        print("  - pytest (testing)")
        print("\nYou can now begin implementing:")
        print("  - Session 1: State Encoding")
        print("  - Session 2: Complete State Encoder")
        print("  - Session 3-5: Neural Network")
        print("  - Session 6-8: MCTS")
    else:
        print("❌ Some core dependencies are missing!")
        print("\nPlease install missing packages:")
        print("  venv\\Scripts\\pip.exe install -r ml\\requirements.txt")
    print("=" * 70)

    return all_checks_passed


if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
