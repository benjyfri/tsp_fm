import wandb
import yaml
import os
import argparse
import sys

# 0. Setup
parser = argparse.ArgumentParser(description="Run WandB Sweep for EGNN.")
parser.add_argument(
    "--config",
    type=str,
    default="sweep_config.yaml",
    help="Path to the YAML configuration file"
)
args = parser.parse_args()

# --- Sanity Checks ---
# 1. Check for the training script
if not os.path.exists('train_egnn.py'):
    print("\n❌ Error: 'train_egnn.py' not found in the current directory.")
    print("   Please ensure you are running this script from the project root.")
    sys.exit(1)

# 2. Check for the config file
if not os.path.exists(args.config):
    raise FileNotFoundError(f"Configuration file not found: {args.config}")

print(f"Loading configuration from: {args.config}")

# 3. Load YAML
with open(args.config, 'r') as file:
    sweep_configuration = yaml.safe_load(file)

# 4. Extract Project Name Safely
try:
    project_name = sweep_configuration["parameters"]["project_name"]["value"]
except KeyError:
    # Fallback if project_name isn't strictly defined in parameters
    project_name = "tsp_FM_EGNN_Sweep"

# 5. Initialize Sweep
# This registers the sweep with WandB servers
sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project=project_name
)

print(f"\n🚀 Sweep Initiated!")
print(f"   ID:      {sweep_id}")
print(f"   Project: {project_name}")
print(f"   Command: wandb agent {os.environ.get('WANDB_ENTITY', 'user')}/{project_name}/{sweep_id}\n")

# 6. Start the Agent
# count=None runs forever until you stop it (or early termination kills it)
# count=20 runs 20 experiments then stops
print("Starting wandb agent...")
wandb.agent(sweep_id, project=project_name, count=20)