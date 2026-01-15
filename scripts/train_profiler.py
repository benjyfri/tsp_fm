#!/usr/bin/env python3
"""
Profiling script to find training bottlenecks
"""
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

from src.models import get_spectral_canonicalization

def profile_model(model, x, t, num_iterations=10):
    """Profile model forward pass"""
    device = next(model.parameters()).device

    # Warmup
    for _ in range(3):
        _ = model(x, t)

    # Time overall
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _ = model(x, t)
    torch.cuda.synchronize()
    avg_time = (time.time() - start) / num_iterations
    print(f"Average forward pass: {avg_time * 1000:.2f}ms")

    # Detailed profiling
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        with record_function("model_forward"):
            for _ in range(5):
                _ = model(x, t)

    # Print results
    print("\n=== Top 10 CPU Time Consumers ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print("\n=== Top 10 CUDA Time Consumers ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\n=== Memory Usage ===")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

    # Export trace for Chrome trace viewer
    prof.export_chrome_trace("trace.json")
    print("\n✓ Trace exported to trace.json (view at chrome://tracing)")


def profile_specific_components(model, x, t):
    """Profile specific model components"""
    print("\n=== Component-Level Profiling ===")

    # Test canonicalization
    if hasattr(model, '__class__') and 'Spectral' in model.__class__.__name__:

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _, _, _, _ = get_spectral_canonicalization(x)
        torch.cuda.synchronize()
        canon_time = (time.time() - start) / 100
        print(f"Spectral canonicalization: {canon_time * 1000:.2f}ms")

        # Test without canonicalization
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            h = model.input_proj(x)
        torch.cuda.synchronize()
        proj_time = (time.time() - start) / 100
        print(f"Input projection only: {proj_time * 1000:.2f}ms")


def measure_data_loading(train_loader):
    """Measure data loading speed"""
    print("\n=== Data Loading Profile ===")

    times = []
    for i, batch in enumerate(train_loader):
        if i >= 20:  # Test first 20 batches
            break
        start = time.time()
        # Simulate moving to GPU
        if isinstance(batch, (list, tuple)):
            _ = [b.cuda() if torch.is_tensor(b) else b for b in batch]
        else:
            _ = batch.cuda()
        times.append(time.time() - start)

    print(f"Average batch loading time: {np.mean(times) * 1000:.2f}ms")
    print(f"Max batch loading time: {np.max(times) * 1000:.2f}ms")


if __name__ == "__main__":
    # Example usage - adapt to your setup
    import sys

    sys.path.append('..')
    from src.models import SpectralCanonTransformer


    # Mock config
    class Config:
        num_points = 50
        embed_dim = 256
        num_heads = 8
        num_layers = 12
        dropout = 0.0


    config = Config()
    model = SpectralCanonTransformer(config).cuda()
    model.eval()

    # Create dummy batch
    B, N = 32, 50
    x = torch.randn(B, N, 2).cuda()
    t = torch.rand(B).cuda()

    print("Starting profiling...")
    profile_model(model, x, t)
    # profile_specific_components(model, x, t)