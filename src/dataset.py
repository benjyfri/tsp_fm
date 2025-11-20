import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data(path, device):
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data at {path}")

    x0_list = [torch.from_numpy(e['points']).float() for e in data]
    x1_list = [torch.from_numpy(e['circle']).float() for e in data]
    theta_list = [e['theta'] for e in data]

    # For calculation of Gap, we need the original edge lengths or original path
    # Storing original path for inference
    paths = [torch.from_numpy(e['path']).long() for e in data]

    x0 = torch.stack(x0_list).to(device)
    x1 = torch.stack(x1_list).to(device)
    theta = torch.tensor(theta_list, dtype=torch.float32).to(device)

    return x0, x1, theta, paths

def get_loader(x0, x1, theta, batch_size, shuffle=True):
    dataset = TensorDataset(x0, x1, theta)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)