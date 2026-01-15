import wandb
import yaml
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20, help="Number of runs to execute")
    parser.add_argument("--entity", type=str, default=None, help="WandB entity/username (optional)")
    args = parser.parse_args()

    # Load the YAML config
    config_path = "sweep_regression.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find {config_path}")

    with open(config_path, 'r') as file:
        sweep_config = yaml.safe_load(file)

    # Extract project name from YAML, fallback to default if missing
    project_name = sweep_config.get('project', 'Angle_Regression_Default')

    print(f"📋 Loaded configuration for project: {project_name}")

    # Initialize the Sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name, entity=args.entity)

    print(f"✅ Sweep created! ID: {sweep_id}")
    print(f"👉 View dashboard at: https://wandb.ai/{args.entity or 'your-username'}/{project_name}/sweeps/{sweep_id}")
    print(f"🚀 Starting agent to run {args.count} experiments...")

    # Start the agent
    wandb.agent(sweep_id, count=args.count, project=project_name, entity=args.entity)


if __name__ == "__main__":
    main()