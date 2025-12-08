import os
import subprocess

# Define experiments: Comparing model capacity
experiments = [
    {
        "name": "Reg_Small",
        "embed_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "batch_size": 256
    },
    {
        "name": "Reg_Medium",
        "embed_dim": 256,
        "num_layers": 8,
        "num_heads": 8,
        "batch_size": 128
    },
    {
        "name": "Reg_Large",
        "embed_dim": 512,
        "num_layers": 12,
        "num_heads": 8,
        "batch_size": 64
    },
]

# Adjust paths as necessary
DATA_TRAIN = "data/processed_data_geom_train.pt"
DATA_VAL = "data/processed_data_geom_val.pt"

for exp in experiments:
    print(f"--- Running {exp['name']} ---")
    cmd = [
        "python", "train_regression.py",
        "--project_name", "tsp-regression-benchmarks",
        "--run_name", exp['name'],
        "--train_data", DATA_TRAIN,
        "--val_data", DATA_VAL,
        "--embed_dim", str(exp['embed_dim']),
        "--num_layers", str(exp['num_layers']),
        "--num_heads", str(exp['num_heads']),
        "--batch_size", str(exp['batch_size']),
        "--epochs", "100"
    ]

    subprocess.run(cmd)