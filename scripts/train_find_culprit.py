#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
import sys

# --- FORCE SETTINGS ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- IMPORT MODEL ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models import SpectralCanonTransformer


class Config:
    """Dummy config to match your training args"""
    num_points = 50
    embed_dim = 512
    num_heads = 8
    num_layers = 12
    dropout = 0.0


def benchmark_matmul(device, size=4096, dtype=torch.float32, use_tf32=True):
    """Benchmarks raw matrix multiplication speed."""
    # Toggle TF32
    torch.backends.cuda.matmul.allow_tf32 = use_tf32

    A = torch.randn(size, size, device=device, dtype=dtype)
    B = torch.randn(size, size, device=device, dtype=dtype)

    # Warmup
    for _ in range(10):
        C = torch.mm(A, B)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(50):
        C = torch.mm(A, B)
    end.record()
    torch.cuda.synchronize()

    avg_time = start.elapsed_time(end) / 50  # ms
    tflops = (2 * size ** 3) / (avg_time * 1e-3) / 1e12

    return avg_time, tflops


def inspect_model():
    print("\n" + "=" * 40)
    print("HARDWARE & PRECISION INSPECTION")
    print("=" * 40)

    if not torch.cuda.is_available():
        print("CRITICAL: CUDA not available.")
        return

    device = torch.device('cuda:3')  # Using the ID from your logs
    props = torch.cuda.get_device_properties(device)
    print(f"GPU Name: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")

    # Check TF32 Support
    # Ampere is Compute Capability 8.0+
    has_tf32 = props.major >= 8
    print(f"Supports TF32: {has_tf32} " + ("" if has_tf32 else ""))

    print("\n--- Speed Benchmark (4096 x 4096 MatMul) ---")

    # 1. Standard FP32 (TF32 Disabled)
    time_fp32, tflops_fp32 = benchmark_matmul(device, use_tf32=False)
    print(f"Standard Float32: {time_fp32:.2f} ms | {tflops_fp32:.2f} TFLOPS")

    # 2. TensorFloat-32 (TF32 Enabled)
    if has_tf32:
        time_tf32, tflops_tf32 = benchmark_matmul(device, use_tf32=True)
        print(f"TensorFloat-32:   {time_tf32:.2f} ms | {tflops_tf32:.2f} TFLOPS")
        print(f"Speedup: {time_fp32 / time_tf32:.2f}x")
    else:
        print("TensorFloat-32:   NOT SUPPORTED (Hardware Limit)")

    print("\n--- Model Inspection ---")

    config = Config()
    # Instantiate as if in training
    model = SpectralCanonTransformer(config).to(device)

    # FORCE FLOAT cast (like in V4 script)
    model = model.float()

    print("Checking Layer Dtypes:")
    has_double = False

    # Recursive check of all parameters and buffers
    for name, param in model.named_parameters():
        if param.dtype == torch.float64:
            print(f"!! FOUND DOUBLE PARAM: {name}")
            has_double = True

    for name, buf in model.named_buffers():
        if buf.dtype == torch.float64:
            print(f"!! FOUND DOUBLE BUFFER: {name}")
            has_double = True

    if not has_double:
        print("✓ All Parameters and Buffers are Float32.")
    else:
        print("CRITICAL: Model contains Float64 components!")


if __name__ == "__main__":
    inspect_model()