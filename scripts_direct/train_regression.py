#!/usr/bin/env python3
"""
train_regression.py
Direct coordinate regression (x0 -> x1) without Flow Matching.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

# --- FIX 1: Add Parent Directory to System Path ---
# This ensures we can import 'src' regardless of where we run the script from
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import models and data loader
from src.models import CanonicalRegressor
from src.dataset import load_data, get_loader

# --- FIX 2: Path Resolution Helper ---
def resolve_path(path_str):
    """
    Tries to find the file in the current directory,
    or inside the project root if running from a subdirectory.
    """
    path = Path(path_str)
    # 1. Check if absolute or exists relative to current execution dir
    if path.exists():
        return path

    # 2. Check relative to project root
    root_path = Path(project_root) / path_str
    if root_path.exists():
        return root_path

    # 3. Return original (will likely fail later, but preserves error message)
    return path

def save_checkpoint(path, model, optimizer=None, epoch=None, args=None, best=False):
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    if args is not None:
        payload['args'] = vars(args)

    torch.save(payload, path)
    if best:
        best_path = str(Path(path).with_name(Path(path).stem + "_best.pt"))
        torch.save(payload, best_path)

def train(args):
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Setup ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Run Name
    if args.run_name is None:
        args.run_name = f"Regression-D{args.embed_dim}-L{args.num_layers}"

    # Re-init wandb to ensure config is captured if not already
    if wandb.run is None:
        wandb.init(project=args.project_name, name=args.run_name, config=args)
    config = wandb.config
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    # --- Load Data ---
    print("Loading data...")
    # FIX: Use resolve_path to find data even if running from scripts/
    train_path = resolve_path(config.train_data)
    val_path = resolve_path(config.val_data)

    print(f"  -> Train: {train_path}")
    print(f"  -> Val:   {val_path}")

    # load_data returns: x0, x1, theta, paths, precomputed
    x0_train, x1_train, theta_train, _, _ = load_data(str(train_path), 'cpu', interpolant=None)
    x0_val, x1_val, theta_val, _, _ = load_data(str(val_path), 'cpu', interpolant=None)

    # Create Loaders
    train_loader = get_loader(x0_train, x1_train, theta_train, precomputed=None,
                              batch_size=config.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    val_loader = get_loader(x0_val, x1_val, theta_val, precomputed=None,
                            batch_size=config.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # --- Initialize Model ---
    print("Initializing Canonical Regressor...")
    # Ensure config object has all necessary attributes for model init
    # (Sometimes wandb config objects behave slightly differently than argparse args)
    model = CanonicalRegressor(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {num_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    save_dir = Path(config.save_dir) / config.run_name
    os.makedirs(save_dir, exist_ok=True)

    print("\nStarting Regression Training...")

    for epoch in range(1, config.epochs + 1):
        # --- Training ---
        model.train()
        train_loss_accum = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", ncols=100)

        for batch in pbar:
            b_x0, b_x1 = batch[0], batch[1]

            b_x0 = b_x0.to(device, non_blocking=True)
            b_x1 = b_x1.to(device, non_blocking=True)

            optimizer.zero_grad()

            pred_x1 = model(b_x0)
            loss = criterion(pred_x1, b_x1)

            loss.backward()

            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            optimizer.step()

            train_loss_accum += loss.item()
            batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        avg_train = train_loss_accum / max(1, batches)

        # --- Validation ---
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                b_x0, b_x1 = batch[0], batch[1]
                b_x0 = b_x0.to(device, non_blocking=True)
                b_x1 = b_x1.to(device, non_blocking=True)

                pred_x1 = model(b_x0)
                val_loss = criterion(pred_x1, b_x1)

                val_loss_accum += val_loss.item()
                val_batches += 1

        avg_val = val_loss_accum / max(1, val_batches)

        scheduler.step()

        # --- Logging ---
        print(f"Epoch {epoch}: Train MSE {avg_train:.6f} | Val MSE {avg_val:.6f}")
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train,
            "val_loss": avg_val,
            "lr": optimizer.param_groups[0]['lr']
        })

        # --- Checkpointing ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_checkpoint(save_dir / "best_model.pt", model, optimizer, epoch, args, best=True)
            print(f" -> New Best Model (Val: {best_val_loss:.6f})")

    # Final Save
    save_checkpoint(save_dir / "final_model.pt", model, optimizer, config.epochs, args)
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--project_name', type=str, default="tsp-regression")
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default="./checkpoints_reg")

    # Model Params
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)

    # Training Params
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    train(args)