import torch
import numpy as np
import sys
import os

# --- 1. Setup Path to find models.py ---
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    # Changed import to the new Regressor model
    from models_regression import EquivariantAngleRegressor
except ImportError:
    print("❌ Could not import EquivariantAngleRegressor.")
    print("Ensure you are running this from the project root and 'src.models' exists.")
    sys.exit(1)


# --- 2. Mock Configuration ---
class TestConfig:
    # Must satisfy: (embed_dim // num_heads) // 2 must be divisible by 4
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

    # 1. Radial (Center distance)
    centroid = x.mean(dim=1, keepdim=True)
    centered = x - centroid
    r = torch.norm(centered, dim=-1)  # (B, N)

    # 2. Angles
    theta = torch.atan2(centered[..., 1], centered[..., 0])
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    # 3. Mock Hull (Just use R max - R as a proxy)
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
    model = EquivariantAngleRegressor(config).to(device)
    model.eval()

    # A. Create Data
    B, N = 2, 50
    x = torch.randn(B, N, 2).to(device)
    signals = generate_mock_signals(x)

    # B. Forward Pass (Original)
    with torch.no_grad():
        # No time input 't' anymore
        out_original = model(x, static_signals=signals)  # Shape (B, N)

    # C. Permute Data
    perm = torch.randperm(N)

    # Apply permutation to inputs AND signals
    x_perm = x[:, perm, :]
    signals_perm = signals[:, perm, :]

    # D. Forward Pass (Permuted)
    with torch.no_grad():
        out_perm_input = model(x_perm, static_signals=signals_perm)

    # E. Check Results
    # Since output is (B, N) scalar angles, we expect:
    # out_perm_input == out_original[:, perm]

    expected_output = out_original[:, perm]

    diff = (out_perm_input - expected_output).abs().max().item()

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out_original.shape} (Expected B, N)")
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
    model = EquivariantAngleRegressor(config).to(device)
    model.eval()

    sizes_to_test = [20, 50, 100, 127]  # Random sizes

    try:
        for n in sizes_to_test:
            # Create batch with size N
            x = torch.randn(2, n, 2).to(device)
            signals = generate_mock_signals(x)

            with torch.no_grad():
                out = model(x, static_signals=signals)

            print(f"N={n:<3} -> Output Shape: {list(out.shape)} ... OK")

            # Expecting (B, N) output
            if out.shape != (2, n):
                raise ValueError(f"Output shape mismatch for N={n}. Expected (2, {n}), got {out.shape}")

        print("✅ PASS: Model handles variable point cloud sizes correctly.")

    except Exception as e:
        print(f"❌ FAIL: Crashed on variable size test. Error: {e}")


def test_rope_frequency_isolation():
    print(f"\n{'=' * 60}")
    print("TEST 3: RoPE Frequency Isolation (Correct Banding)")
    print(f"{'=' * 60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TestConfig()
    model = EquivariantAngleRegressor(config).to(device)
    model.eval()

    B, N = 1, 32
    x = torch.randn(B, N, 2).to(device)
    signals = generate_mock_signals(x)

    # Compute reference frequencies
    freqs_ref = model.compute_frequencies(signals)

    signal_names = ["radius", "sinθ", "cosθ", "hull"]

    for i, name in enumerate(signal_names):
        modified = signals.clone()
        modified[..., i] *= 1.1  # Perturb one signal

        freqs_mod = model.compute_frequencies(modified)

        # Check difference
        diff = (freqs_mod - freqs_ref).abs().mean().item()
        print(f"Signal '{name:<6}': mean |Δfreqs| = {diff:.4e}")

        # Ensure this signal actually affects the encoding
        assert diff > 1e-6, f"❌ Signal '{name}' does not affect RoPE frequencies."

    print("✅ PASS: Each signal independently controls RoPE phase.")


if __name__ == "__main__":
    test_permutation_equivariance()
    test_variable_size()
    test_rope_frequency_isolation()