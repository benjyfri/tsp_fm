#!/usr/bin/env python3
"""
Revised train.py - Optimized for L40S/Ampere GPUs.
Fixes: Float64 poisoning, enables TF32, optimizes data casting.
"""

import os

# --- 1. Prevent Geomstats Poisoning ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
# --------------------------------------

import sys
import argparse
from pathlib import Path
import math
import time
import numpy as np
import torch
import wandb
from tqdm import tqdm

# --- 2. CRITICAL OPTIMIZATION: Enable TensorFloat-32 ---
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# -----------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.geometry import GeometryProvider
from src.interpolants import get_interpolant
from src.models import (
    VectorFieldModel, RoPEVectorFieldModel, CanonicalMLPVectorField,
    CanonicalRoPEVectorField, CanonicalRegressor,
    SpectralCanonMLP,
    SpectralCanonTransformer,
    EquivariantDiffTransformer
)
from src.dataset import load_data, get_loader


def get_model(args, device):
    """Factory for model creation with explicit Float32 cast."""
    if args.model_type == 'rope':
        model = RoPEVectorFieldModel(args)
    elif args.model_type == 'canonical_rope':
        model = CanonicalRoPEVectorField(args)
    elif args.model_type == 'canonical_mlp':
        model = CanonicalMLPVectorField(args)
    elif args.model_type == 'canonical_regressor':
        model = CanonicalRegressor(args)
    elif args.model_type == 'spectral_mlp':
        model = SpectralCanonMLP(args)
    elif args.model_type == 'spectral_trans':
        model = SpectralCanonTransformer(args)
    elif args.model_type == 'equivariant_transformer':
        model = EquivariantDiffTransformer(args)
    else:
        model = VectorFieldModel(args)

    # CRITICAL: Force cast to float32 to override any geomstats defaults
    return model.to(device).float()


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, args=None, best=False):
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    if optimizer is not None: payload['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None: payload['scheduler_state_dict'] = scheduler.state_dict()
    if args is not None: payload['args'] = vars(args)

    torch.save(payload, path)
    if best:
        best_path = str(Path(path).with_name(Path(path).stem + "_best.pt"))
        torch.save(payload, best_path)


def train(args):
    # --- 3. CRITICAL OPTIMIZATION: Reset Global Dtype ---
    torch.set_default_dtype(torch.float32)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup WandB
    # 1. Calculate parameter count first
    _temp_model = get_model(args, device='cpu')
    num_params = sum(p.numel() for p in _temp_model.parameters()) / 1e6
    print(f'+++++++++++++++++++++++++++++++')
    print(f'Num of parameters: {num_params}')
    print(f'+++++++++++++++++++++++++++++++')
    del _temp_model

    # 2. Construct dynamic run name if one isn't manually provided
    if args.run_name is None:
        # Format: Type_Params_Layers_Heads_Dim_LR
        # Example: spectral_trans_2.5M_L12_H8_D512_lr1e-04
        args.run_name = (
            f"{args.model_type}"
            f"_{num_params:.2f}M"  # Number of params (e.g. 12.50M)
            f"_L{args.num_layers}"  # Layers
            f"_H{args.num_heads}"  # Heads
            f"_D{args.embed_dim}"  # Embed Dimension
            f"_lr{args.lr:.0e}"  # Learning rate in scientific notation
        )

    # 3. Initialize WandB with the new name
    wandb.init(project=args.project_name, name=args.run_name, config=args)

    # 4. Explicitly log param count to config so you can sort/filter by "num_params_M" in the Table UI
    wandb.config.update({"num_params_M": num_params}, allow_val_change=True)
    config = wandb.config
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Path handling
    def resolve_path(path_str):
        p = Path(path_str)
        return p if p.is_absolute() else Path(parent_dir) / p

    save_dir = resolve_path(config.save_dir) / config.run_name
    os.makedirs(save_dir, exist_ok=True)

    # Geometry & Data
    geo = GeometryProvider(config.num_points)
    interpolant = get_interpolant(config.interpolant, geo, stochasticity_scale=config.stochasticity_scale)

    print("Loading data...")
    try:
        ### CHANGED: Unpack 5 values now (x0, x1, paths, signals, precomputed)
        # We ignore paths (index 2) using '_'
        x0, x1, _, signals_train, pre_train = load_data(str(resolve_path(config.train_data)), 'cpu',
                                                        interpolant=interpolant)
        x0_val, x1_val, _, signals_val, pre_val = load_data(str(resolve_path(config.val_data)), 'cpu',
                                                            interpolant=interpolant)
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: {e}")
        sys.exit(1)

    ### CHANGED: Pass signals to get_loader
    train_loader = get_loader(x0, x1, signals_train, pre_train, batch_size=config.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = get_loader(x0_val, x1_val, signals_val, pre_val, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Model Setup
    model = get_model(config, device)
    model = model.float()  # Double-check cast

    # --- 5. OPTIMIZATION: Compile Model (PyTorch 2.0+) ---
    # This fuses kernels for RoPE and Attention
    print("Compiling model...")
    try:
        model = torch.compile(model, mode='default')
    except Exception as e:
        print(f"Could not compile model: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    eta_min = config.lr * config.eta_min_factor
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=eta_min)

    best_val = float('inf')
    use_geo = geo if 'kendall' in config.interpolant else None

    print("\nStarting training (High Performance Mode)")
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", ncols=120)

        for batch in pbar:
            ### NEW: Robust Batch Unpacking Logic
            # 1. Base components (Always first 2)
            b_x0, b_x1 = batch[0], batch[1]
            idx_counter = 2

            # 2. Check for Static Signals
            b_signals = None
            if signals_train is not None:
                b_signals = batch[idx_counter].to(device, non_blocking=True)
                idx_counter += 1

            # 3. Check for Precomputed Interpolant Data
            precomputed_batch = ()
            if pre_train is not None:
                # All remaining items are precomputed
                precomputed_batch = tuple(
                    t.to(device, dtype=torch.float32, non_blocking=True) for t in batch[idx_counter:])

            # Cast inputs
            b_x0 = b_x0.to(device, dtype=torch.float32, non_blocking=True)
            b_x1 = b_x1.to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()

            # Sample (Ensure interpolant returns floats)
            t, xt, ut = interpolant.sample(b_x0, b_x1, *precomputed_batch, device=device)
            xt, t, ut = xt.float(), t.float(), ut.float()

            ### CHANGED: Conditional Forward Pass
            if args.model_type == 'equivariant_transformer':
                if b_signals is None:
                    raise ValueError("Model requires static signals but dataset didn't provide them.")
                vt = model(xt, t, static_signals=b_signals, geometry=use_geo)
            else:
                # Standard models don't take static_signals
                vt = model(xt, t, geometry=use_geo)
            loss = torch.mean((vt - ut) ** 2)

            # --- ADD THIS BLOCK ---
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n❌ CRITICAL: Loss is {loss.item()}. Stopping run immediately.")

                # Optional: Mark run as failed in WandB explicitly
                wandb.run.summary["status"] = "crashed_nan"
                wandb.finish(exit_code=1)

                # Exit with error code 1 so the Sweep Agent knows it failed
                sys.exit(1)
            # ----------------------

            loss.backward()

            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train = epoch_train_loss / max(1, num_batches)

        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                # (Repeat unpacking logic for validation)
                b_x0, b_x1 = batch[0], batch[1]
                idx_counter = 2

                b_signals = None
                if signals_val is not None:
                    b_signals = batch[idx_counter].to(device, non_blocking=True)
                    idx_counter += 1

                precomputed_batch = ()
                if pre_val is not None:
                    precomputed_batch = tuple(
                        t.to(device, dtype=torch.float32, non_blocking=True) for t in batch[idx_counter:])

                b_x0 = b_x0.to(device, dtype=torch.float32, non_blocking=True)
                b_x1 = b_x1.to(device, dtype=torch.float32, non_blocking=True)

                t, xt, ut = interpolant.sample(b_x0, b_x1, *precomputed_batch, device=device)
                xt, t, ut = xt.float(), t.float(), ut.float()

                ### CHANGED: Conditional Forward Pass (Validation)
                if args.model_type == 'equivariant_transformer':
                    vt = model(xt, t, static_signals=b_signals, geometry=use_geo)
                else:
                    vt = model(xt, t, geometry=use_geo)

                val_loss_accum += torch.mean((vt - ut) ** 2).item()
                val_batches += 1

        avg_val = val_loss_accum / max(1, val_batches)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        print(f"Epoch {epoch}: Train {avg_train:.6f} | Val {avg_val:.6f} | LR {current_lr:.3e}")
        wandb.log({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val, "lr": current_lr})

        # Checkpointing
        if config.checkpoint_freq > 0 and (epoch % config.checkpoint_freq == 0):
            save_checkpoint(os.path.join(save_dir, "last_checkpoint.pt"), model, optimizer, scheduler, epoch, args)

        if config.checkpoint_freq > 0 and (avg_val < best_val):
            best_val = avg_val
            save_checkpoint(os.path.join(save_dir, "best_model.pt"), model, optimizer, scheduler, epoch, args,
                            best=True)
            print(f"  -> New best model saved (val {best_val:.6f})")

    save_checkpoint(os.path.join(save_dir, "final_model.pt"), model, optimizer, scheduler, config.epochs, args)
    print(f"Training complete. Best validation loss: {best_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flow matching model")

    # Args from your original script
    parser.add_argument('--project_name', type=str, default="tsp_FM")
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--interpolant', type=str,
                        choices=['kendall', 'linear', 'angle', 'kendall_sfm', 'withguard_kendall', 'linear_sfm'],
                        default='kendall')
    parser.add_argument('--stochasticity_scale', type=float, default=0.1)
    ### CHANGED: added equivariant_transformer to choices
    parser.add_argument('--model_type', type=str, default='rope',
                        choices=['concat', 'rope', 'canonical_mlp', 'canonical_rope',
                                 'canonical_regressor', 'spectral_mlp', 'spectral_trans',
                                 'equivariant_transformer'])
    parser.add_argument('--save_dir', type=str, default="./checkpoints")
    parser.add_argument('--checkpoint_freq', type=int, default=10)
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eta_min_factor', type=float, default=1e-3)

    # --- 7. OPTIMIZATION: Disable Gradient Clipping by Default ---
    # Clipping forces a CPU sync. Only enable if training is unstable.
    parser.add_argument('--grad_clip_norm', type=float, default=0.0)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gpu_id', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Force float32 in args just in case
    torch.set_default_dtype(torch.float32)

    train(args)