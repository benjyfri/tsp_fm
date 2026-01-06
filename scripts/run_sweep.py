import wandb
import yaml
import os
import argparse

# 0. Parse command-line arguments
parser = argparse.ArgumentParser(description="Run WandB Sweep from a specific YAML configuration.")
parser.add_argument(
    "--config",
    type=str,
    default="sweep_config.yaml",
    help="Path to the YAML configuration file (default: sweep_config.yaml)"
)
args = parser.parse_args()

# Ensure the scripts folder exists (sanity check)
if not os.path.exists('scripts/train.py'):
    print("Warning: scripts/train.py not found. The agent might fail if the script is missing.")

# Check if the config file exists before trying to open it
if not os.path.exists(args.config):
    raise FileNotFoundError(f"Configuration file not found: {args.config}")

print(f"Loading configuration from: {args.config}")

# 1. Load the YAML configuration safely
with open(args.config, 'r') as file:
    sweep_configuration = yaml.safe_load(file)

# 2. Initialize the sweep
# Login if needed (wandb.login())
sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project="tsp_FM"
)

print(f"Sweep initiated. ID: {sweep_id}")

# 3. Run the agent automatically
# 'count' limits the number of runs. Remove it to run forever.
print("Starting wandb agent to run the sweep...")
wandb.agent(sweep_id, project="tsp_FM", count=30)