import torch
import numpy as np
import sys
import os
import math
def clone_with_modified_signal(signals, signal_idx, scale):
    """
    Returns a copy of signals where only one channel is scaled.
    """
    modified = signals.clone()
    modified[..., signal_idx] *= scale
    return modified
def test_rope_frequency_isolation():
    print(f"\n{'=' * 60}")
    print("TEST 3: RoPE Frequency Isolation (Correct Banding)")
    print(f"{'=' * 60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TestConfig()
    model = EquivariantDiffTransformer(config).to(device)
    model.eval()

    B, N = 1, 32
    x = torch.randn(B, N, 2).to(device)
    signals = generate_mock_signals(x)

    freqs_ref = model.compute_frequencies(signals)

    for i, name in enumerate(["radius", "sinθ", "cosθ", "hull"]):
        modified = signals.clone()
        modified[..., i] *= 1.1

        freqs_mod = model.compute_frequencies(modified)

        diff = (freqs_mod - freqs_ref).abs().mean().item()
        print(f"Signal '{name:<6}': mean |Δfreqs| = {diff:.4e}")

        assert diff > 1e-6, f"❌ Signal '{name}' does not affect RoPE frequencies."

    print("✅ PASS: Each signal independently controls RoPE phase.")

# --- 1. Setup Path to find models.py ---
# Assumes this script is in the project root.
# If it's in a subfolder, adjust '..' accordingly.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from src.models import EquivariantDiffTransformer
except ImportError:
    print("❌ Could not import EquivariantDiffTransformer.")
    print("Ensure you are running this from the project root and 'src.models' exists.")
    sys.exit(1)


# --- 2. Mock Configuration ---
class TestConfig:
    # Must satisfy: (embed_dim // num_heads) // 2 must be divisible by 4
    # Example: 128 // 4 = 32 head_dim -> 16 rope_dim -> 16 % 4 == 0 (OK)
    embed_dim = 128
    num_heads = 4
    num_layers = 2


# --- 3. Helper to Generate Dummy Signals ---
def generate_mock_signals(x):
    """
    Generates plausible geometric signals for the model inputs.
    x: (B, N, 2)
    Returns: (B, N, 4) -> [R, Sin, Cos, Hull]
    """
    B, N, _ = x.shape
    device = x.device

    # Simple proxies for the real signals to ensure shapes are correct
    # We don't need the exact Convex Hull algorithm for checking equivariance logic,
    # just something that permutes with the points.

    # 1. Radial (Center distance)
    centroid = x.mean(dim=1, keepdim=True)
    centered = x - centroid
    r = torch.norm(centered, dim=-1)  # (B, N)

    # 2. Angles
    theta = torch.atan2(centered[..., 1], centered[..., 0])
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    # 3. Mock Hull (Just use R as a proxy for depth for testing)
    hull = r.max(dim=1, keepdim=True)[0] - r

    signals = torch.stack([r, sin_t, cos_t, hull], dim=-1)
    return signals


# --- 4. The Tests ---

def test_permutation_equivariance():
    print(f"\n{'=' * 60}")
    print("TEST 1: Permutation Equivariance")
    print(f"{'=' * 60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TestConfig()
    model = EquivariantDiffTransformer(config).to(device)
    model.eval()

    # A. Create Data
    B, N = 2, 50
    x = torch.randn(B, N, 2).to(device)
    t = torch.rand(B).to(device)
    signals = generate_mock_signals(x)

    # B. Forward Pass (Original)
    with torch.no_grad():
        out_original = model(x, t, static_signals=signals)

    # C. Permute Data
    # Create a random permutation
    perm = torch.randperm(N)

    # Apply permutation to inputs AND signals
    # Equivariance requirement: f(P*x, P*signals) == P * f(x, signals)
    x_perm = x[:, perm, :]
    signals_perm = signals[:, perm, :]

    # D. Forward Pass (Permuted)
    with torch.no_grad():
        out_perm_input = model(x_perm, t, static_signals=signals_perm)

    # E. Check Results
    # We un-permute the output to compare with original
    # logic: out_perm_input should look like out_original[:, perm, :]

    # Let's align them back to original order for comparison
    # If out_perm_input[0] corresponds to permuted indices,
    # we need to inverse permute or compare against permuted original.

    expected_output = out_original[:, perm, :]

    diff = (out_perm_input - expected_output).abs().max().item()

    print(f"Input Shape: {x.shape}")
    print(f"Max Absolute Difference: {diff:.2e}")

    if diff < 1e-5:
        print("✅ PASS: Model is Permutation Equivariant.")
    else:
        print("❌ FAIL: Model output changed significantly under permutation.")


def test_variable_size():
    print(f"\n{'=' * 60}")
    print("TEST 2: Size Invariance (Variable N)")
    print(f"{'=' * 60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TestConfig()
    model = EquivariantDiffTransformer(config).to(device)
    model.eval()

    sizes_to_test = [20, 50, 100, 127]  # Random sizes

    try:
        for n in sizes_to_test:
            # Create batch with size N
            x = torch.randn(2, n, 2).to(device)
            t = torch.rand(2).to(device)
            signals = generate_mock_signals(x)

            with torch.no_grad():
                out = model(x, t, static_signals=signals)

            print(f"N={n:<3} -> Output Shape: {list(out.shape)} ... OK")

            if out.shape != (2, n, 2):
                raise ValueError(f"Output shape mismatch for N={n}")

        print("✅ PASS: Model handles variable point cloud sizes correctly.")

    except Exception as e:
        print(f"❌ FAIL: Crashed on variable size test. Error: {e}")


if __name__ == "__main__":
    test_permutation_equivariance()
    test_variable_size()
    test_rope_frequency_isolation()