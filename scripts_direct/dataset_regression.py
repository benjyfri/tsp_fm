import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data(path, device, interpolant=None):
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data at {path}")

    x0_list = [torch.as_tensor(e['points']).float() for e in data]
    x1_list = [torch.as_tensor(e['circle']).float() for e in data]
    paths = [torch.as_tensor(e['path']).long() for e in data]

    x0 = torch.stack(x0_list)
    x1 = torch.stack(x1_list)

    # --- Check for Static RoPE Signals ---
    static_signals = None
    if 'static_signals' in data[0]:
        print("Found precomputed static RoPE signals.")
        sig_list = [torch.as_tensor(e['static_signals']) for e in data]
        static_signals = torch.stack(sig_list).float()  # (Total, N, 4)
    else:
        print("No static RoPE signals found.")

    # Kendall Normalization
    is_kendall = False
    if interpolant is not None:
        # Check if it's a class or string
        if isinstance(interpolant, str):
            is_kendall = 'kendall' in interpolant.lower()
        else:
            class_name = interpolant.__class__.__name__.lower()
            is_kendall = 'kendall' in class_name

    if is_kendall:
        print(f"Kendall-type interpolant detected. Normalizing coordinates...")

        def shape_normalize(x):
            x_d = x.double()
            x_centered = x_d - x_d.mean(dim=1, keepdim=True)
            norm = torch.norm(x_centered, p='fro', dim=(1, 2), keepdim=True)
            return (x_centered / (norm + 1e-10)).float()

        x0 = shape_normalize(x0)
        x1 = shape_normalize(x1)
    else:
        x0 = x0.float()
        x1 = x1.float()

    # --- NEW: Convert GT X1 (Circle Coords) to Angles ---
    # 1. Center the target circle data to ensure atan2 is valid
    x1_center = x1.mean(dim=1, keepdim=True)
    x1_centered = x1 - x1_center

    # 2. Compute angles [-pi, pi]
    # x1_centered shape: (B, N, 2) -> angles shape: (B, N)
    x1_angles = torch.atan2(x1_centered[..., 1], x1_centered[..., 0])

    print("Converted GT circle coordinates to angles (radians).")

    # --- Precompute Logic (Optional, usually for Flow Matching) ---
    # Note: If you are doing pure regression, interpolant might be None.
    precomputed = None
    if interpolant is not None and hasattr(interpolant, 'precompute'):
        print(f"Pre-computing data for {interpolant.__class__.__name__}...")
        # Note: Interpolants usually expect Coords. If your interpolant
        # relies on x1 being coordinates, this might break.
        # But for direct regression, interpolant is usually None.
        precomputed_raw = interpolant.precompute(x0, x1)
        if precomputed_raw is not None:
            precomputed = precomputed_raw
            for k, v in list(precomputed.items()):
                if isinstance(v, (list, tuple)):
                    precomputed[k] = [torch.as_tensor(x) for x in v]
                else:
                    precomputed[k] = torch.as_tensor(v) if not torch.is_tensor(v) else v

    # Return x1_angles instead of x1 coords
    return x0, x1_angles, paths, static_signals, precomputed


def get_loader(x0, x1, static_signals=None, precomputed=None, batch_size=256, shuffle=True, num_workers=4,
               pin_memory=True):
    # x1 is now angles (B, N)
    tensors = [x0, x1]

    # If static_signals exist, add them as 3rd element
    if static_signals is not None:
        tensors.append(static_signals)

    if precomputed is not None:
        tensors = tensors + list(precomputed.values())

    dataset = TensorDataset(*tensors)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)