#!/usr/bin/env python3
"""
Revised train.py with:
- geomstats backend set early
- reproducible seeding (numpy, torch, geomstats)
- AdamW with weight decay
- CosineAnnealingLR scheduler (eta_min = lr * 1e-3)
- optional linear warmup
- gradient clipping
- full checkpoint saving (model + optimizer + scheduler + args)
- saving best model
- wandb logging + local checkpointing
"""

import os
# --- MUST set geomstats backend before any geomstats import (or imports that import geomstats) ---
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

# Add parent dir to path to import src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import local modules (these may import geomstats; backend already set above)
from src.geometry import GeometryProvider
from src.interpolants import get_interpolant
from src.models import VectorFieldModel
from src.dataset import load_data, get_loader

# Now import geomstats backend for seeding (works because GEOMSTATS_BACKEND already set)
try:
    import geomstats.backend as gs
except Exception:
    gs = None  # If geomstats not available here, seed call will be skipped gracefully


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, args=None, best=False):
    """
    Save a checkpoint including optimizer and scheduler state for full resume capability.
    If best==True, save to '<path>_best.pt' as well.
    """
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


def linear_warmup_lr(optimizer, base_lr, warmup_steps, step):
    """Set the optimizer's lr to linear warmup value (in-place)."""
    if warmup_steps <= 0:
        return
    lr_mult = min(1.0, float(step) / float(max(1, warmup_steps)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * lr_mult


def train(args):
    # seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if gs is not None:
        try:
            gs.random.seed(args.seed)
        except Exception:
            pass

    # Build descriptive run name if not provided
    if args.run_name is None:
        args.run_name = f"{args.interpolant}-N{args.num_points}-ep{args.epochs}-lr{args.lr:g}"

    # Initialize wandb
    wandb.init(project=args.project_name, name=args.run_name, config=args)
    config = wandb.config

    # Set metric axis in wandb
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    # device
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        try:
            print(f"  -> GPU: {torch.cuda.get_device_name(config.gpu_id)}")
            # Optionally set default device (not required if moving tensors explicitly)
            # torch.set_default_device(device)
        except Exception:
            pass

    # --- Path handling helper (resolve relative to repo root) ---
    def resolve_path(path_str):
        p = Path(path_str)
        if not p.is_absolute():
            return Path(parent_dir) / p
        return p

    train_path = resolve_path(config.train_data)
    val_path = resolve_path(config.val_data)

    # Setup save directory
    save_dir = resolve_path(config.save_dir) / config.run_name
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints and models will be saved to: {save_dir}")

    print(f"Loading training data from: {train_path}")

    # Geometry provider & interpolant
    geo = GeometryProvider(config.num_points)
    interpolant = get_interpolant(config.interpolant, geo)

    # Load data (dataset loader may already place tensors on device)
    try:
        x0, x1, theta, _ = load_data(str(train_path), device)
        x0_val, x1_val, theta_val, _ = load_data(str(val_path), device)
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: {e}")
        sys.exit(1)

    train_loader = get_loader(x0, x1, theta, config.batch_size, shuffle=True)
    val_loader = get_loader(x0_val, x1_val, theta_val, config.batch_size, shuffle=False)

    # Build model & optimizer
    model = VectorFieldModel(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Scheduler: CosineAnnealingLR across epochs with stronger final anneal
    eta_min = config.lr * config.eta_min_factor  # recommended default 1e-3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=eta_min
    )

    # Training bookkeeping
    best_val = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    use_geo = geo if config.interpolant == 'kendall' else None

    total_steps = 0
    warmup_steps = int(config.warmup_epochs)  # warmup in epochs; we will call per-epoch linear warmup

    print("\nStarting training")
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", ncols=100)
        # If warmup requested: set lr multiplier per-epoch (simple linear warmup across warmup_epochs)
        if warmup_steps > 0 and epoch <= warmup_steps:
            # compute linear multiplier between 0..1
            lr_mult = float(epoch) / float(max(1, warmup_steps))
            for pg in optimizer.param_groups:
                pg['lr'] = config.lr * lr_mult

        for b_x0, b_x1, b_theta in pbar:
            b_x0 = b_x0.to(device)
            b_x1 = b_x1.to(device)
            b_theta = b_theta.to(device)

            optimizer.zero_grad()

            # Sample path (interpolant does double-upcast internally if implemented)
            t, xt, ut = interpolant.sample(b_x0, b_x1, b_theta, device)

            # Forward
            vt = model(xt, t, geometry=use_geo)

            loss = torch.mean((vt - ut) ** 2)
            loss.backward()

            # Gradient clipping
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1
            total_steps += 1

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

        # Step scheduler once per epoch (after validation)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['lr'].append(current_lr)

        # Logging
        print(f"Epoch {epoch}: Train {avg_train:.6f} | Val {avg_val:.6f} | LR {current_lr:.3e}")
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train,
            "val_loss": avg_val,
            "lr": current_lr
        }, step=epoch)

        # Checkpointing: periodic and best
        # Full checkpoint with optimizer & scheduler state
        if config.checkpoint_freq > 0 and (epoch % config.checkpoint_freq == 0):
            ckpt_path = os.path.join(save_dir, "last_checkpoint.pt")
            save_checkpoint(ckpt_path, model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, args=args)

        # Save best model
        if config.checkpoint_freq > 0 and (avg_val < best_val):
            best_val = avg_val
            best_path = os.path.join(save_dir, "best_model.pt")
            save_checkpoint(best_path, model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, args=args, best=True)
            print(f"  -> New best model saved (val {best_val:.6f})")

    # Final save (also store training history)
    final_path = os.path.join(save_dir, "final_model.pt")
    save_checkpoint(final_path, model, optimizer=optimizer, scheduler=scheduler, epoch=config.epochs, args=args)
    # also save simple JSON history for convenience
    try:
        import json
        hist_path = os.path.join(save_dir, "train_history.json")
        with open(hist_path, "w") as fh:
            json.dump(history, fh)
    except Exception:
        pass

    # Save a copy to wandb run dir
    try:
        wandb_save_path = os.path.join(wandb.run.dir, "final_model.pt")
        save_checkpoint(wandb_save_path, model, optimizer=optimizer, scheduler=scheduler, epoch=config.epochs, args=args)
    except Exception:
        pass

    print(f"Training complete. Final model saved to {final_path}")
    print(f"Best validation loss: {best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flow matching model (revised)")

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
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eta_min_factor', type=float, default=1e-3,
                        help="Scheduler eta_min will be lr * eta_min_factor (recommended 1e-3)")
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help="Number of initial epochs to linearly warm up the LR (default 0)")
    parser.add_argument('--grad_clip_norm', type=float, default=5.0,
                        help="Gradient clipping norm (<=0 disables clipping)")

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set PyTorch default dtype (keeps behaviour consistent)
    torch.set_default_dtype(torch.float32)

    train(args)
