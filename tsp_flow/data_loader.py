import torch
import torch.utils.data
import numpy as np
import os

import torch
import numpy as np

class PointCloudDataset(torch.utils.data.Dataset):
    """
    Loads the processed .pt file (Kendall format).
    Each item returns the (x0, x1, theta, gt_length) tuple.
    """
    def __init__(self, data_file):
        try:
            self.entries = torch.load(data_file, weights_only=False)
            print(f"Successfully loaded {len(self.entries)} total data entries from {data_file}")
        except FileNotFoundError:
            print(f"Error: File not found at {data_file}. Please check the path.")
            self.entries = []
        except Exception as e:
            print(f"Error loading file: {e}")
            self.entries = []

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # The flow starts at 'points' (x0) and ends at 'circle' (x1)
        x0 = torch.from_numpy(entry['points'].astype(np.float32))
        x1 = torch.from_numpy(entry['circle'].astype(np.float32))

        # Get the theta (Procrustes distance) scalar
        theta = torch.tensor(entry['theta'], dtype=torch.float32)

        # [FIXED] Add ground-truth TSP length for evaluation.
        # Use .get() for safety, providing 0.0 as a default if 'total_length' is missing.
        # This is required by evaluate.py but ignored by train.py.
        gt_length = torch.tensor(entry.get('total_length', 0.0), dtype=torch.float32)

        return x0, x1, theta, gt_length

def get_loaders(train_data_file, test_data_file, batch_size):
    """
    Creates and returns the training and test DataLoader objects
    from pre-split data files.
    """
    try:
        # Load train_dataset from train_data_file
        print(f"Loading training data from: {train_data_file}")
        train_dataset = PointCloudDataset(train_data_file)
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty. Check train_data_file path.")

        # Load test_dataset from test_data_file
        print(f"Loading test data from: {test_data_file}")
        test_dataset = PointCloudDataset(test_data_file)
        if len(test_dataset) == 0:
            raise ValueError("Test dataset is empty. Check test_data_file path.")

        print(f"Loaded {len(train_dataset)} train samples and {len(test_dataset)} test samples.")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Adjust num_workers based on your system
            pin_memory=True,
            persistent_workers=True if os.name != 'nt' else False # persistent_workers for speed, if not on Windows
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4, # Adjust num_workers based on your system
            pin_memory=True,
            persistent_workers=True if os.name != 'nt' else False
        )

        # Return test_dataset for use in evaluate.py
        return train_loader, test_loader, test_dataset

    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return None, None, None