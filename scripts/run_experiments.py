import os
import subprocess

# Define experiments
experiments = [
    {"name": "Exp_Kendall", "interpolant": "kendall", "epochs": 50},
    {"name": "Exp_Linear", "interpolant": "linear", "epochs": 50},
    {"name": "Exp_Angle", "interpolant": "angle", "epochs": 50},
]

DATA_TRAIN = "data/processed_data_geom_train.pt"
DATA_VAL = "data/processed_data_geom_val.pt"

for exp in experiments:
    print(f"--- Running {exp['name']} ---")
    cmd = [
        "python", "scripts/train.py",
        "--project_name", "tsp-fm-comparison",
        "--train_data", DATA_TRAIN,
        "--val_data", DATA_VAL,
        "--interpolant", exp['interpolant'],
        "--epochs", str(exp['epochs'])
    ]

    subprocess.run(cmd)