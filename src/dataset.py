# dataset.py
import torch
from torch.utils.data import TensorDataset, DataLoader


# Import GeometryProvider if you want to enable the normalization fix
# from src.geometry import GeometryProvider

def load_data(path, device, interpolant=None):
    """
    Args:
        interpolant: Optional. If provided, will pre-compute interpolant-specific data.
    """
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data at {path}")

    # --- Robust Data Conversion ---
    # Works regardless of whether data was saved as numpy arrays or torch tensors
    x0_list = [torch.as_tensor(e['points']).float() for e in data]
    x1_list = [torch.as_tensor(e['circle']).float() for e in data]

    # Handle paths (Long/Int64)
    paths = [torch.as_tensor(e['path']).long() for e in data]

    x0 = torch.stack(x0_list)  # float32
    x1 = torch.stack(x1_list)  # float32

    # Only normalize if we are working in Kendall Shape Space
    # Check if 'kendall' is in the class name of the interpolant object
    is_kendall = False
    if interpolant is not None:
        class_name = interpolant.__class__.__name__.lower()
        is_kendall = 'kendall' in class_name

    if is_kendall:
        print(f"Kendall-type interpolant ({interpolant.__class__.__name__}) detected.")
        print("Projecting data to Shape Space (Centering + Frobenius Norm=1)...")

        def shape_normalize(x):
            x_d = x.double()
            # Centering: Subtract mean of points (B, N, 2)
            x_centered = x_d - x_d.mean(dim=1, keepdim=True)
            # Normalizing: Frobenius norm = 1 (B, 1, 1)
            norm = torch.norm(x_centered, p='fro', dim=(1, 2), keepdim=True)
            return (x_centered / (norm + 1e-10)).float()

        x0 = shape_normalize(x0)
        x1 = shape_normalize(x1)
    else:
        x0 = x0.float()
        x1 = x1.float()

    # Let interpolant pre-compute if it wants to
    precomputed = None
    if interpolant is not None and hasattr(interpolant, 'precompute'):
        print(f"Pre-computing data for {interpolant.__class__.__name__}...")
        precomputed_raw = interpolant.precompute(x0, x1)

        # --- FIX: Check if precomputed_raw is None before iterating ---
        if precomputed_raw is not None:
            precomputed = precomputed_raw  # Assign to the variable we return

            # IMPORTANT: do NOT cast precomputed tensors to float here if they are double.
            for k, v in list(precomputed.items()):
                if isinstance(v, (list, tuple)):
                    precomputed[k] = [torch.as_tensor(x) for x in v]
                else:
                    precomputed[k] = torch.as_tensor(v) if not torch.is_tensor(v) else v
        else:
            # If interpolant returns None (like Linear), ensure precomputed remains None
            precomputed = None

    return x0, x1, paths, precomputed


def get_loader(x0, x1, precomputed=None, batch_size=256, shuffle=True, num_workers=4, pin_memory=True):
    """
    Creates a DataLoader. Default uses num_workers=4 and pin_memory=True.
    """
    if precomputed is not None:
        # Unpack all precomputed tensors and add to dataset
        # Ensure precomputed values are sorted/deterministic if needed,
        # though .values() usually respects insertion order in modern Python.
        tensors = [x0, x1] + list(precomputed.values())
        dataset = TensorDataset(*tensors)
    else:
        dataset = TensorDataset(x0, x1)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)