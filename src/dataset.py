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

    x0_list = [torch.from_numpy(e['points']).float() for e in data]
    x1_list = [torch.from_numpy(e['circle']).float() for e in data]
    paths = [torch.from_numpy(e['path']).long() for e in data]

    x0 = torch.stack(x0_list)  # float32
    x1 = torch.stack(x1_list)  # float32

    # --- RECOMMENDED FIX FOR MANIFOLD DRIFT ---
    # Uncomment the lines below to force training data onto the sphere (Norm=1).
    # This prevents the "Distribution Shift" between Training (High Norm) and Inference (Unit Norm).
    # ------------------------------------------
    # num_points = x0.shape[1]
    # geo = GeometryProvider(num_points)
    # x0 = geo.space.projection(x0)
    # x1 = geo.space.projection(x1)
    # ------------------------------------------

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