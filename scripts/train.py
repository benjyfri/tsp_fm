#!/usr/bin/env python3
"""
Revised train.py with support for Canonical models.
"""

import os
# --- MUST set geomstats backend before any geomstats import ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
# ----------------------------------------------------------------

import sys
import argparse
from pathlib import Path
import math
import time

import numpy as np
import torch
import wandb
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import all model variants
from src.geometry import GeometryProvider
from src.interpolants import get_interpolant
from src.models import (
    VectorFieldModel,
    RoPEVectorFieldModel,
    CanonicalMLPVectorField,
    CanonicalRoPEVectorField
)
from src.dataset import load_data, get_loader

try:
    import geomstats.backend as gs
except Exception:
    gs = None

def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, args=None, best=False):
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        payload['scheduler_state_dict'] = scheduler.state_dict()
    if args is not None:
        payload['args'] = vars(args)

    torch.save(payload, path)
    if best:
        best_path = str(Path(path).with_name(Path(path).stem + "_best.pt"))
        torch.save(payload, best_path)

def get_model(args, device):
    """Helper to instantiate the correct model based on args.model_type"""
    if args.model_type == 'rope':
        print("Initializing RoPE Vector Field Model...")
        return RoPEVectorFieldModel(args).to(device)
    elif args.model_type == 'canonical_rope':
        print("Initializing Canonical RoPE Vector Field Model...")
        return CanonicalRoPEVectorField(args).to(device)
    elif args.model_type == 'canonical_mlp':
        print("Initializing Canonical MLP Vector Field Model...")
        return CanonicalMLPVectorField(args).to(device)
    else:
        print("Initializing Standard Concatenation Vector Field Model...")
        return VectorFieldModel(args).to(device)

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if gs is not None:
        try:
            gs.random.seed(args.seed)
        except Exception:
            pass

    # --- Compute Parameter Count for Run Name ---
    # Instantiate temp model on CPU
    _temp_model = get_model(args, device='cpu')
    num_params = sum(p.numel() for p in _temp_model.parameters())
    params_m = num_params / 1e6
    del _temp_model

    if args.run_name is None:
        args.run_name = f"{args.model_type}-{args.interpolant}-D{args.embed_dim}-L{args.num_layers}-P{params_m:.1f}M"

    wandb.init(project=args.project_name, name=args.run_name, config=args)
    config = wandb.config

    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Path handling
    def resolve_path(path_str):
        p = Path(path_str)
        if not p.is_absolute():
            return Path(parent_dir) / p
        return p

    train_path = resolve_path(config.train_data)
    val_path = resolve_path(config.val_data)
    save_dir = resolve_path(config.save_dir) / config.run_name
    os.makedirs(save_dir, exist_ok=True)

    # Geometry & Data
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

    # Instantiate Model
    model = get_model(config, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    eta_min = config.lr * config.eta_min_factor
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=eta_min
    )

    best_val = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    # Pass geometry if using Kendall interpolant
    use_geo = geo if config.interpolant == 'kendall' else None

    warmup_steps = int(config.warmup_epochs)

    print("\nStarting training")
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", ncols=100)

        # Simple Linear Warmup
        if warmup_steps > 0 and epoch <= warmup_steps:
            lr_mult = float(epoch) / float(max(1, warmup_steps))
            for pg in optimizer.param_groups:
                pg['lr'] = config.lr * lr_mult

        for b_x0, b_x1, b_theta in pbar:
            b_x0 = b_x0.to(device)
            b_x1 = b_x1.to(device)
            b_theta = b_theta.to(device)

            optimizer.zero_grad()

            # Sample Interpolant
            t, xt, ut = interpolant.sample(b_x0, b_x1, b_theta, device)

            # Forward
            vt = model(xt, t, geometry=use_geo)

            loss = torch.mean((vt - ut) ** 2)
            loss.backward()

            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        avg_train = epoch_train_loss / max(1, num_batches)

        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for b_x0, b_x1, b_theta in val_loader:
                b_x0 = b_x0.to(device)
                b_x1 = b_x1.to(device)
                b_theta = b_theta.to(device)

                t, xt, ut = interpolant.sample(b_x0, b_x1, b_theta, device)
                vt = model(xt, t, geometry=use_geo)
                val_loss_accum += torch.mean((vt - ut) ** 2).item()
                val_batches += 1

        avg_val = val_loss_accum / max(1, val_batches)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # History & Logging
        print(f"Epoch {epoch}: Train {avg_train:.6f} | Val {avg_val:.6f} | LR {current_lr:.3e}")
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train,
            "val_loss": avg_val,
            "lr": current_lr
        }, step=epoch)

        # Checkpointing
        if config.checkpoint_freq > 0 and (epoch % config.checkpoint_freq == 0):
            save_checkpoint(os.path.join(save_dir, "last_checkpoint.pt"),
                            model, optimizer, scheduler, epoch, args)

        if config.checkpoint_freq > 0 and (avg_val < best_val):
            best_val = avg_val
            save_checkpoint(os.path.join(save_dir, "best_model.pt"),
                            model, optimizer, scheduler, epoch, args, best=True)
            print(f"  -> New best model saved (val {best_val:.6f})")

    # Final Save
    save_checkpoint(os.path.join(save_dir, "final_model.pt"),
                    model, optimizer, scheduler, config.epochs, args)

    print(f"Training complete. Best validation loss: {best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flow matching model")

    parser.add_argument('--project_name', type=str, default="tsp-flow-matching")
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--interpolant', type=str, choices=['kendall', 'linear', 'angle'], default='kendall')

    # Updated Model Choices
    parser.add_argument('--model_type', type=str, default='rope',
                        choices=['concat', 'rope', 'canonical_mlp', 'canonical_rope'],
                        help="Choose model architecture")

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
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eta_min_factor', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--grad_clip_norm', type=float, default=5.0)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    train(args)