#!/usr/bin/env python3
"""
Angle Regression Train Script (Vector-based)
Optimized for Ampere+ GPUs (L40S).
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

# --- Optimizations ---
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Ensure we import the NEW models
from models_regression import EquivariantAngleRegressor
from dataset_regression import load_data, get_loader


def get_model(args, device):
    model = EquivariantAngleRegressor(args)
    return model.to(device).float()


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, args=None, best=False):
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    if optimizer: payload['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler: payload['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(payload, path)
    if best:
        best_path = str(Path(path).with_name(Path(path).stem + "_best.pt"))
        torch.save(payload, best_path)


# --- NEW: Robust Geometric Loss ---
def vector_phase_loss(pred_vector, target_angle):
    """
    Optimizes the vector to point at the correct angle on the unit circle.

    pred_vector: (B, N, 2) -> Raw unnormalized predictions
    target_angle: (B, N) -> GT Angles in radians
    """
    # 1. Project predictions onto the unit circle
    # eps=1e-6 prevents NaN gradients if the model outputs (0,0)
    pred_norm = F.normalize(pred_vector, p=2, dim=-1, eps=1e-6)

    # 2. Convert GT angles to Unit Vectors
    gt_sin = torch.sin(target_angle)
    gt_cos = torch.cos(target_angle)
    gt_vector = torch.stack([gt_cos, gt_sin], dim=-1)  # (B, N, 2)

    # 3. MSE on the manifold (Unit Circle)
    return F.mse_loss(pred_norm, gt_vector)


def train(args):
    # Reset Dtype
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Setup WandB & Params ---
    _temp_model = get_model(args, device='cpu')
    num_params = sum(p.numel() for p in _temp_model.parameters()) / 1e6
    del _temp_model

    if args.run_name is None:
        args.run_name = f"VecReg_{num_params:.2f}M_L{args.num_layers}_H{args.num_heads}_lr{args.lr:.0e}"

    wandb.init(project=args.project_name, name=args.run_name, config=args)
    wandb.config.update({"num_params_M": num_params})

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Params: {num_params:.2f}M")

    save_dir = Path(args.save_dir) / args.run_name
    os.makedirs(save_dir, exist_ok=True)

    # --- Data Loading ---
    print("Loading data...")
    # NOTE: load_data returns (x0, x1_angle, path, signals, precomputed)
    # We only care about x0, x1, and signals for regression
    x0, x1, _, signals_train, _ = load_data(args.train_data, 'cpu', interpolant=None)
    x0_val, x1_val, _, signals_val, _ = load_data(args.val_data, 'cpu', interpolant=None)

    train_loader = get_loader(x0, x1, signals_train, None, batch_size=args.batch_size, shuffle=True, num_workers=8,
                              pin_memory=True)
    val_loader = get_loader(x0_val, x1_val, signals_val, None, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            pin_memory=True)

    # --- Model Setup ---
    model = get_model(args, device)

    print("Compiling model...")
    try:
        model = torch.compile(model, mode='reduce-overhead')
    except Exception as e:
        print(f"Compile skipped: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

    best_val = float('inf')

    print("\nStarting Training (Vector Phase Loss)")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_acc = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100)

        for batch in pbar:
            # --- ROBUST UNPACKING ---
            # Batch structure depends on dataset.py
            # Expected: [x0, x1, path, signals] OR [x0, x1, signals]
            b_coords = batch[0].to(device, non_blocking=True)
            b_angles_gt = batch[1].to(device, non_blocking=True)

            # Find signals in the remaining elements
            b_signals = None
            if len(batch) >= 3:
                # If signals are the last element, pick them
                # Commonly index 3 if path is included, or 2 if not
                b_signals = batch[-1].to(device, non_blocking=True)

            if b_signals is None:
                raise ValueError("Could not find static_signals in batch! Check dataset loader.")

            optimizer.zero_grad(set_to_none=True)

            # Model predicts (B, N, 2) vector
            pred_vectors = model(b_coords, b_signals)

            # Loss projects to unit circle and compares
            loss = vector_phase_loss(pred_vectors, b_angles_gt)

            if torch.isnan(loss):
                print("NaN Loss detected");
                sys.exit(1)

            loss.backward()

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            optimizer.step()

            train_loss_acc += loss.item()
            batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train = train_loss_acc / max(1, batches)

        # --- Validation ---
        model.eval()
        val_loss_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                b_coords = batch[0].to(device, non_blocking=True)
                b_angles_gt = batch[1].to(device, non_blocking=True)
                b_signals = batch[-1].to(device, non_blocking=True)

                pred_vectors = model(b_coords, b_signals)
                val_loss_acc += vector_phase_loss(pred_vectors, b_angles_gt).item()
                val_batches += 1

        avg_val = val_loss_acc / max(1, val_batches)
        scheduler.step()

        print(f"Epoch {epoch}: Train {avg_train:.5f} | Val {avg_val:.5f} | LR {optimizer.param_groups[0]['lr']:.2e}")
        wandb.log({"train_loss": avg_train, "val_loss": avg_val, "epoch": epoch})

        if avg_val < best_val:
            best_val = avg_val
            save_checkpoint(save_dir / "best_model.pt", model, optimizer, scheduler, epoch, best=True)
            print(f"  -> New best model (Val: {best_val:.5f})")

    save_checkpoint(save_dir / "final_model.pt", model, optimizer, scheduler, args.epochs)
    print(f"Done. Best Val: {best_val:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default="./checkpoints")
    parser.add_argument('--project_name', type=str, default="Angle_Regression")
    parser.add_argument('--run_name', type=str, default=None)

    # Model Args
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)

    # Training Args
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grad_clip_norm', type=float, default=0.0)

    args = parser.parse_args()
    train(args)