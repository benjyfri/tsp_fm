import torch
import torch.utils.data
import numpy as np
import os

import torch
import numpy as np

class PointCloudDataset(torch.utils.data.Dataset):
    """
    Loads the 'processed_tsp_dataset.pt' file.
    Each item returns the (noisy_input, tar_circle, total_length) tuple.
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

        # Get the total_length (scalar) and convert to a tensor
        total_length = torch.tensor(entry['total_length'], dtype=torch.float32)

        return x0, x1, total_length

def get_loaders(data_file, batch_size, train_split_size=15000, seed=42):
    """
    Creates and returns the training and test DataLoader objects.
    """
    try:
        full_dataset = PointCloudDataset(data_file)
        if len(full_dataset) == 0:
            raise ValueError("Dataset is empty. Check data_file path.")

        test_size = len(full_dataset) - train_split_size
        if test_size <= 0:
             raise ValueError(f"Full dataset size ({len(full_dataset)}) is not larger than TRAIN_SIZE ({train_split_size}).")

        print(f"Splitting data into {train_split_size} train and {test_size} test samples.")

        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_split_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, test_loader, test_dataset

    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return None, None, None