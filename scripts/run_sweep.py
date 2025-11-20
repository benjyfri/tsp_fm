import wandb
import os

# Login to wandb (make sure you have run 'wandb login' in terminal previously)
# wandb.login()

sweep_id = wandb.sweep(
    sweep=wandb.load_yaml("sweep_config.yaml"),
    project="tsp-flow-matching"
)

print(f"Sweep initiated. ID: {sweep_id}")
print("Run the agent with:")
print(f"wandb agent {sweep_id}")