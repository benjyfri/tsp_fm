import argparse
import torch
import wandb
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent dir to path to import src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.geometry import GeometryProvider
from src.interpolants import get_interpolant
from src.models import VectorFieldModel
from src.dataset import load_data, get_loader

def save_checkpoint(model, args, epoch, path):
    """
    Saves model weights AND hyperparameters (args) to a single file.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'args': vars(args)  # Converts Namespace to dict
    }
    torch.save(checkpoint, path)

def train(args):
    # --- FIX: Descriptive Run Name ---
    if args.run_name is None:
        args.run_name = f"{args.interpolant}-N{args.num_points}-ep{args.epochs}"

    # 1. Setup WandB
    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=args
    )
    config = wandb.config

    # Set default x-axis to epoch
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")

    # --- Path Handling ---
    def resolve_path(path_str):
        p = Path(path_str)
        if not p.is_absolute():
            return Path(parent_dir) / p
        return p

    train_path = resolve_path(config.train_data)
    val_path = resolve_path(config.val_data)

    # 2. Setup Local Save Directory
    save_dir = resolve_path(config.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints and models will be saved to: {save_dir}")

    print(f"Loading training data from: {train_path}")

    # 3. Prepare Geometry & Data
    geo = GeometryProvider(config.num_points)
    interpolant = get_interpolant(config.interpolant, geo)

    try:
        x0, x1, theta, _ = load_data(str(train_path), device)
        x0_val, x1_val, theta_val, _ = load_data(str(val_path), device)
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: {e}")
        sys.exit(1)

    train_loader = get_loader(x0, x1, theta, config.batch_size, shuffle=True)
    val_loader = get_loader(x0_val, x1_val, theta_val, config.batch_size, shuffle=False)

    # 4. Model
    model = VectorFieldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # 5. Loop
    use_geo = geo if config.interpolant == 'kendall' else None

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for b_x0, b_x1, b_theta in pbar:
            optimizer.zero_grad()

            # Sample path
            t, xt, ut = interpolant.sample(b_x0, b_x1, b_theta, device)

            # Predict
            vt = model(xt, t, geometry=use_geo)

            loss = torch.mean((vt - ut)**2)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_x0, b_x1, b_theta in val_loader:
                t, xt, ut = interpolant.sample(b_x0, b_x1, b_theta, device)
                vt = model(xt, t, geometry=use_geo)
                val_loss += torch.mean((vt - ut)**2).item()

        avg_val = val_loss / len(val_loader)

        # Log
        # wandb log is enough, pbar handles the live view
        wandb.log({"train_loss": avg_train, "val_loss": avg_val, "epoch": epoch})

        # 1. Checkpoint Saving (using helper function)
        if config.checkpoint_freq > 0 and (epoch + 1) % config.checkpoint_freq == 0:
            ckpt_name = f"checkpoint_ep{epoch+1}.pt"
            save_checkpoint(model, config, epoch, os.path.join(save_dir, ckpt_name))

    # 2. Save Final Model Locally
    final_save_path = os.path.join(save_dir, "final_model.pt")
    save_checkpoint(model, config, config.epochs, final_save_path)

    # Also save to wandb dir
    wandb_save_path = os.path.join(wandb.run.dir, "model.pt")
    save_checkpoint(model, config, config.epochs, wandb_save_path)

    print(f"Training complete. Model saved to {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default="tsp-flow-matching")
    parser.add_argument('--run_name', type=str, default=None)

    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--interpolant', type=str, choices=['kendall', 'linear', 'angle'], default='kendall')

    parser.add_argument('--save_dir', type=str, default="./checkpoints")
    parser.add_argument('--checkpoint_freq', type=int, default=10)

    # Hyperparams
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gpu_id', type=int, default=7)

    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)
    train(args)