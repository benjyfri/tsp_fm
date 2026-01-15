import torch
import numpy as np
import sys
import os
import warnings

# --- 1. Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress TF32 warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 2. Import your Model ---
try:
    # Adjust this if your training script is named differently
    from train_egnn import SparseEGNNFlowMatching, get_knn_graph
except ImportError:
    print("⚠️  Could not import model.")
    print("    Please ensure train_egnn.py is in the python path.")
    sys.exit(1)


# ============================================================================
#  TEST SUITE
# ============================================================================

def run_trials(test_name, test_func, num_trials=10):
    print(f"\n{'=' * 60}")
    print(f"{test_name}")
    print(f"{'=' * 60}")

    passed = 0
    total_diff = 0.0

    # Handle tests that don't return a diff (like variable size)
    for i in range(num_trials):
        result = test_func()

        # Unpack result based on length
        if isinstance(result, tuple):
            diff, is_pass = result
            total_diff += diff
        else:
            is_pass = result
            diff = 0.0

        if is_pass:
            passed += 1
        else:
            print(f"   Trial {i + 1} FAILED. Diff: {diff:.2e}")

    if passed == num_trials:
        msg = f"✅ PASS: {passed}/{num_trials} trials successful."
        if total_diff > 0: msg += f" (Avg Diff: {total_diff / num_trials:.2e})"
        print(msg)
    else:
        print(f"❌ FAIL: Only {passed}/{num_trials} trials passed.")


def check_permutation():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Use small k so it works for small N=20
    model = SparseEGNNFlowMatching(hidden_dim=32, depth=2, k=5).to(device)
    model.eval()

    B, N = 2, 20
    x = torch.randn(B, N, 2).to(device)
    t = torch.rand(B).to(device)

    # Original
    edge_index_orig = get_knn_graph(x, model.k)
    with torch.no_grad():
        v_orig = model(x, t, edge_index_orig)

    # Permute
    perm = torch.randperm(N)
    x_perm = x[:, perm, :]

    # Recompute Graph & Forward
    edge_index_perm = get_knn_graph(x_perm, model.k)
    with torch.no_grad():
        v_perm = model(x_perm, t, edge_index_perm)

    # Un-permute output
    v_perm_unshuffled = torch.zeros_like(v_perm)
    v_perm_unshuffled[:, perm, :] = v_perm

    diff = (v_perm_unshuffled - v_orig).abs().max().item()
    return diff, diff < 1e-5


def check_rotation():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SparseEGNNFlowMatching(hidden_dim=32, depth=2, k=5).to(device)
    model.eval()

    B, N = 2, 20
    x = torch.randn(B, N, 2).to(device)
    t = torch.rand(B).to(device)

    # Random Rotation Matrix
    theta = np.random.rand() * 2 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    Q = torch.tensor([[c, -s], [s, c]], dtype=torch.float32).to(device)

    # Original
    edge_index_orig = get_knn_graph(x, model.k)
    with torch.no_grad():
        v_orig = model(x, t, edge_index_orig)

    # Rotate Input
    x_rot = x @ Q.T

    # Rotated Forward
    edge_index_rot = get_knn_graph(x_rot, model.k)
    with torch.no_grad():
        v_rot = model(x_rot, t, edge_index_rot)

    # Compare: Rotate original output to match
    v_expected = v_orig @ Q.T

    diff = (v_rot - v_expected).abs().max().item()
    return diff, diff < 1e-4  # Slightly looser tolerance for general rotation float error


def check_translation():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SparseEGNNFlowMatching(hidden_dim=32, depth=2, k=5).to(device)
    model.eval()

    B, N = 2, 20
    x = torch.randn(B, N, 2).to(device)
    t = torch.rand(B).to(device)

    # Original
    edge_index_orig = get_knn_graph(x, model.k)
    with torch.no_grad():
        v_orig = model(x, t, edge_index_orig)

    # Random Shift
    shift = torch.randn(1, 1, 2).to(device) * 100
    x_shifted = x + shift

    # Shifted Forward
    edge_index_shift = get_knn_graph(x_shifted, model.k)
    with torch.no_grad():
        v_shifted = model(x_shifted, t, edge_index_shift)

    # Compare (Should be identical)
    diff = (v_shifted - v_orig).abs().max().item()
    return diff, diff < 1e-5


def check_variable_size():
    """
    Tests that the SAME model instance can handle batches with different N
    without crashing or shape mismatch errors.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Initialize model ONCE
    k_neighbors = 10
    model = SparseEGNNFlowMatching(hidden_dim=32, depth=2, k=k_neighbors).to(device)
    model.eval()

    # List of sizes to test in this single trial
    # Include an odd number (127) to catch reshaping bugs
    sizes_to_test = [20, 50, 100, 127]

    all_passed = True

    for N in sizes_to_test:
        B = 2
        try:
            x = torch.randn(B, N, 2).to(device)
            t = torch.rand(B).to(device)

            # 1. Compute Graph (Dynamic Size)
            edge_index = get_knn_graph(x, model.k)

            # 2. Forward Pass
            with torch.no_grad():
                v = model(x, t, edge_index)

            # 3. Check Output Shape
            if v.shape != (B, N, 2):
                print(f"   Size mismatch for N={N}. Expected {(B, N, 2)}, got {v.shape}")
                all_passed = False

        except Exception as e:
            print(f"   Crashed on N={N}: {e}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print(f"Running Geometric Tests...")

    run_trials("TEST 1: Permutation Equivariance", check_permutation)
    run_trials("TEST 2: Rotation Equivariance", check_rotation)
    run_trials("TEST 3: Translation Invariance", check_translation)
    # Run fewer trials for size check as it iterates internally
    run_trials("TEST 4: Variable Graph Size (Inference)", check_variable_size, num_trials=5)