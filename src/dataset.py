# dataset.py
import torch
from torch.utils.data import TensorDataset, DataLoader

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
    theta_list = [e['theta'] for e in data]
    paths = [torch.from_numpy(e['path']).long() for e in data]

    x0 = torch.stack(x0_list)         # float32
    x1 = torch.stack(x1_list)         # float32
    theta = torch.tensor(theta_list, dtype=torch.float32)

    # Let interpolant pre-compute if it wants to
    precomputed = None
    if interpolant is not None and hasattr(interpolant, 'precompute'):
        print(f"Pre-computing data for {interpolant.__class__.__name__}...")
        precomputed = interpolant.precompute(x0, x1, theta)

        # IMPORTANT: do NOT cast precomputed tensors to float here.
        # KendallInterpolant.precompute returns double tensors and sample()
        # expects double precomputed tensors to avoid per-batch .double() calls.
        # If you have any precomputed entries that are numpy arrays, convert them to tensors here
        # (but keep dtype=torch.double where appropriate).
        for k, v in list(precomputed.items()):
            if isinstance(v, (list, tuple)):
                precomputed[k] = [torch.as_tensor(x) for x in v]
            else:
                precomputed[k] = torch.as_tensor(v) if not torch.is_tensor(v) else v

    return x0, x1, theta, paths, precomputed


def get_loader(x0, x1, theta, precomputed=None, batch_size=256, shuffle=True, num_workers=4, pin_memory=True):
    """
    Creates a DataLoader. Default uses num_workers=4 and pin_memory=True.
    """
    if precomputed is not None:
        # Unpack all precomputed tensors and add to dataset
        tensors = [x0, x1, theta] + list(precomputed.values())
        dataset = TensorDataset(*tensors)
    else:
        dataset = TensorDataset(x0, x1, theta)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)
