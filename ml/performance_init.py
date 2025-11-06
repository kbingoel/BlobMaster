"""
Performance initialization module for PyTorch training and inference.

Call init_performance() at startup to enable TF32 and other optimizations.
"""

import torch


def init_performance():
    """
    Initialize performance optimizations for PyTorch training and inference.

    Enables TF32 (TensorFloat-32) on Ampere+ GPUs for faster matrix multiplication
    without significant loss in accuracy. TF32 uses 10-bit mantissa (vs FP32's 23-bit)
    but keeps FP32's 8-bit exponent, providing ~8x speedup on compatible hardware.

    This function should be called once at startup, before any model creation
    or training begins.

    Example:
        >>> from ml.performance_init import init_performance
        >>> init_performance()
        >>> # Now create models and start training
    """
    # Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx/40xx, A100, etc.)
    # TF32 provides ~8x speedup with minimal accuracy loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    print("[Performance] TF32 enabled for faster matrix multiplication")
    print(f"[Performance] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Performance] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Performance] CUDA version: {torch.version.cuda}")
