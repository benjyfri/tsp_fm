import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data(path, device, interpolant=None):
    try:
        # Load data (map to CPU initially to save GPU memory during setup)
        data = torch.load(path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data at {path}")

    # =========================================================
    # 1. Format Detection & Unpacking
    # =========================================================

    # CASE A: New Efficient Format (Dictionary of Tensors)
    if isinstance(data, dict):
        print(f"  -> Loading optimized tensor dictionary from {path}...")
        x0 = data['points'].float()
        x1 = data['circle'].float()
        # 'path' is optional for training but good to have
        paths = data['path'].long() if 'path' in data else None

        if 'static_signals' in data:
            print("  -> Found precomputed static RoPE signals.")
            static_signals = data['static_signals'].float()
        else:
            print("  -> No static RoPE signals found.")
            static_signals = None

    # CASE B: Legacy Format (List of Dictionaries)
    elif isinstance(data, list):
        print(f"  -> Loading legacy list-of-dicts from {path}...")
        x0 = torch.stack([torch.as_tensor(e['points']) for e in data]).float()
        x1 = torch.stack([torch.as_tensor(e['circle']) for e in data]).float()
        paths = torch.stack([torch.as_tensor(e['path']) for e in data]).long()

        if 'static_signals' in data[0]:
            print("  -> Found precomputed static RoPE signals.")
            static_signals = torch.stack([torch.as_tensor(e['static_signals']) for e in data]).float()
        else:
            print("  -> No static RoPE signals found.")
            static_signals = None

    else:
        raise ValueError(f"Unknown data format in {path}: {type(data)}")

    # =========================================================
    # 2. Normalization (If using Kendall)
    # =========================================================
    is_kendall = False
    if interpolant is not None:
        class_name = interpolant.__class__.__name__.lower()
        is_kendall = 'kendall' in class_name

    if is_kendall:
        print(f"  -> Kendall-type interpolant detected. Normalizing...")

        def shape_normalize(x):
            x_d = x.double()
            x_centered = x_d - x_d.mean(dim=1, keepdim=True)
            norm = torch.norm(x_centered, p='fro', dim=(1, 2), keepdim=True)
            return (x_centered / (norm + 1e-10)).float()

        x0 = shape_normalize(x0)
        x1 = shape_normalize(x1)

    # =========================================================
    # 3. Interpolant Precomputation
    # =========================================================
    # We standardize precomputed data into a tuple of tensors
    precomputed_tuple = ()

    if interpolant is not None and hasattr(interpolant, 'precompute'):
        print(f"  -> Pre-computing data for {interpolant.__class__.__name__}...")
        precomputed_raw = interpolant.precompute(x0, x1)

        if precomputed_raw is not None:
            # Handle Dictionary outputs
            if isinstance(precomputed_raw, dict):
                precomputed_tuple = tuple(precomputed_raw.values())
            # Handle List/Tuple outputs
            elif isinstance(precomputed_raw, (list, tuple)):
                precomputed_tuple = tuple(precomputed_raw)
            # Handle Single Tensor output
            else:
                precomputed_tuple = (precomputed_raw,)

            # Ensure everything in the tuple is a Tensor
            precomputed_tuple = tuple(
                torch.as_tensor(v) if not torch.is_tensor(v) else v
                for v in precomputed_tuple
            )

    return x0, x1, paths, static_signals, precomputed_tuple


def get_loader(x0, x1, static_signals=None, precomputed_tuple=(), batch_size=256, shuffle=True, num_workers=4,
               pin_memory=True):
    """
    Constructs a DataLoader.
    Arguments:
      x0, x1: Tensors (B, N, 2)
      static_signals: Optional Tensor (B, N, 4) or None
      precomputed_tuple: Tuple of Tensors (any extra args needed for interpolant)
    """
    tensors = [x0, x1]

    # Add signals if available
    if static_signals is not None:
        tensors.append(static_signals)

    # Add precomputed items if available
    if precomputed_tuple:
        tensors.extend(list(precomputed_tuple))

    dataset = TensorDataset(*tensors)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)