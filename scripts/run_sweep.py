import wandb
import yaml
import os

# Ensure the scripts folder exists (sanity check)
if not os.path.exists('scripts/train.py'):
    print("Warning: scripts/train.py not found. The agent might fail if the script is missing.")

# 1. Load the YAML configuration safely
with open("sweep_config.yaml", 'r') as file:
    sweep_configuration = yaml.safe_load(file)

# 2. Initialize the sweep
# Login if needed (wandb.login())
sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project="tsp-flow-matching"
)

print(f"Sweep initiated. ID: {sweep_id}")

# 3. Run the agent automatically
# 'count' limits the number of runs. Remove it to run forever.
print("Starting wandb agent to run the sweep...")
wandb.agent(sweep_id, project="tsp-flow-matching", count=10)