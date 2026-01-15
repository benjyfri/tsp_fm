#!/usr/bin/env python3
"""
Revised train.py - L40S SPECIAL EDITION.
Optimized for 48GB VRAM + Ada Lovelace Architecture.
Features:
- Massive Throughput (200k Token Budget)
- BFloat16 Precision
- Multi-threaded Data Loading
- Virtual Epochs (Fast Feedback)
"""

import os
import sys
import argparse
from pathlib import Path
import math
import random
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

# --- 1. L40S Optimization Flags ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
# Expandable segments prevents "out of memory" due to fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Enable Ampere/Ada Tensor Cores
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.geometry import GeometryProvider
from src.interpolants import get_interpolant
from src.models import (
    VectorFieldModel, RoPEVectorFieldModel, SpectralCanonTransformer,
    EquivariantDiffTransformer
)
from src.dataset import load_data, get_loader


# ==========================================
# Infinite Loader (Virtual Epochs)
# ==========================================

class InfiniteLoader:
    def __init__(self, loaders):
        self.loaders = loaders
        self.sizes = [len(l) for l in loaders]
        self.total_batches = sum(self.sizes)
        self._reset_iterators()

    def _reset_iterators(self):
        self.iterators = [iter(l) for l in self.loaders]
        self.batch_indices = []
        for loader_idx, num_batches in enumerate(self.sizes):
            self.batch_indices.extend([loader_idx] * num_batches)
        random.shuffle(self.batch_indices)
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= len(self.batch_indices):
            self._reset_iterators()
            self.current_idx = 0

        loader_idx = self.batch_indices[self.current_idx]
        self.current_idx += 1

        try:
            return next(self.iterators[loader_idx])
        except StopIteration:
            self.iterators[loader_idx] = iter(self.loaders[loader_idx])
            return next(self.iterators[loader_idx])

    def __len__(self):
        return self.total_batches


def create_loader_optimized(path, args, interpolant, shuffle=True):
    # Load to CPU
    x0, x1, _, signals, pre = load_data(path.strip(), 'cpu', interpolant=interpolant)

    # --- L40S SCALING ---
    N = x0.shape[1]
    # Target: 200,000 tokens per batch (Utilizes ~30-40GB VRAM)
    TOKEN_BUDGET = 200000

    optimal_bs = int(TOKEN_BUDGET / N)
    # Hard cap at 4096 to prevent CPU bottleneck overhead
    safe_bs = max(4, min(4096, optimal_bs))

    # num_workers=8 is CRITICAL for keeping the GPU fed
    loader = get_loader(x0, x1, signals, pre, batch_size=safe_bs, shuffle=shuffle,
                        num_workers=8, pin_memory=True)

    info = {"name": Path(path).name, "N": N, "bs": safe_bs}
    return loader, info


# ==========================================
# Model Setup
# ==========================================

def get_model(args, device):
    if args.model_type == 'spectral_trans':
        model = SpectralCanonTransformer(args)
    elif args.model_type == 'equivariant_transformer':
        model = EquivariantDiffTransformer(args)
    else:
        model = VectorFieldModel(args)
    return model.to(device).float()


def save_checkpoint(path, model, optimizer, scheduler, epoch, args, best=False):
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args)
    }
    torch.save(payload, path)
    if best:
        torch.save(payload, str(Path(path).with_name(Path(path).stem + "_best.pt")))


def check_model_health(model):
    for param in model.parameters():
        if torch.isnan(param).any(): return False
    return True


# ==========================================
# Training Loop
# ==========================================

def train(args):
    # 1. Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    _temp = get_model(args, 'cpu')
    num_params = sum(p.numel() for p in _temp.parameters()) / 1e6
    del _temp

    if args.run_name is None:
        args.run_name = f"L40S_{args.model_type}_{num_params:.2f}M"

    wandb.init(project=args.project_name, name=args.run_name, config=args)
    # Force X-Axis to be Epochs
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    config = wandb.config

    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")

    save_dir = Path(config.save_dir) / config.run_name
    os.makedirs(save_dir, exist_ok=True)
    last_ckpt_path = os.path.join(save_dir, "last_checkpoint.pt")

    # 2. Data
    geo = GeometryProvider(config.num_points)
    interpolant = get_interpolant(config.interpolant, geo, stochasticity_scale=config.stochasticity_scale)

    print(f"\n{'DATASET':<30} | {'N':<5} | {'Batch Size':<10}")
    print("-" * 50)
    train_paths = config.train_data.split(',')
    train_loaders = []
    for p in train_paths:
        l, info = create_loader_optimized(p.strip(), args, interpolant, True)
        train_loaders.append(l)
        print(f"{info['name']:<30} | {info['N']:<5} | {info['bs']:<10}")
    print("-" * 50)

    # Infinite Training Loader
    infinite_train_loader = InfiniteLoader(train_loaders)

    # Validation Loaders (Standard List)
    val_loaders = [create_loader_optimized(p.strip(), args, interpolant, False)[0] for p in config.val_data.split(',')]

    model = get_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Scheduler: Based on VIRTUAL epochs (e.g. 300 virtual epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 1e-3)

    best_val = float('inf')
    use_geo = geo if 'kendall' in config.interpolant else None

    print(f"\n🚀 L40S ENGINE STARTED")
    print(f"   - Virtual Epoch: {args.steps_per_epoch} steps")
    print(f"   - Precision: BFloat16 Mixed")
    print(f"   - Target Device: {device}")

    step_iterator = iter(infinite_train_loader)

    # 3. Training Loop
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0

        # Updates progress bar every 10 seconds to keep logs clean
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Ep {epoch}", ncols=90, mininterval=10.0)

        for _ in pbar:
            batch = next(step_iterator)

            b_x0, b_x1 = batch[0], batch[1]
            b_signals = None
            precomputed = []
            for i in range(2, len(batch)):
                item = batch[i]
                if b_signals is None and item.dim() == 3 and item.shape[-1] == 4:
                    b_signals = item.to(device, non_blocking=True)
                else:
                    precomputed.append(item.to(device, dtype=torch.float32, non_blocking=True))

            b_x0, b_x1 = b_x0.to(device, non_blocking=True), b_x1.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            try:
                # --- BF16 Context ---
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    t, xt, ut = interpolant.sample(b_x0, b_x1, *tuple(precomputed), device=device)
                    # We cast inputs to float inside the model if needed, but interpolant returns floats.

                    if args.model_type == 'equivariant_transformer':
                        vt = model(xt, t, static_signals=b_signals, geometry=use_geo)
                    else:
                        vt = model(xt, t, geometry=use_geo)

                    loss = torch.mean((vt - ut) ** 2)

                if torch.isnan(loss):
                    if not check_model_health(model):
                        print("💀 Weights NaN. Stopping.")
                        sys.exit(1)
                    optimizer.zero_grad()
                    continue

                loss.backward()

                if config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

                optimizer.step()
                epoch_loss += loss.item()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            except torch.cuda.OutOfMemoryError:
                print(f"[OOM]", end="", flush=True)
                torch.cuda.empty_cache()
                continue

        avg_train = epoch_loss / args.steps_per_epoch

        # --- Validation (BF16) ---
        model.eval()
        val_accum = 0.0
        val_count = 0
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for vl in val_loaders:
                    for batch in vl:
                        b_x0, b_x1 = batch[0].to(device), batch[1].to(device)

                        # Handle extra args (signals/precomputed) for validation
                        b_sig = None
                        b_pre = []
                        for i in range(2, len(batch)):
                            item = batch[i]
                            if b_sig is None and item.dim() == 3 and item.shape[-1] == 4:
                                b_sig = item.to(device)
                            else:
                                b_pre.append(item.to(device))

                        t, xt, ut = interpolant.sample(b_x0, b_x1, *tuple(b_pre), device=device)

                        if args.model_type == 'equivariant_transformer':
                            vt = model(xt, t, static_signals=b_sig, geometry=use_geo)
                        else:
                            vt = model(xt, t, geometry=use_geo)

                        val_accum += torch.mean((vt - ut) ** 2).item()
                        val_count += 1

        avg_val = val_accum / max(1, val_count)
        scheduler.step()

        print(f"Ep {epoch} | Train: {avg_train:.5f} | Val: {avg_val:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        wandb.log({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val, "lr": optimizer.param_groups[0]['lr']})

        # Save Checkpoints
        if epoch % 5 == 0 or avg_val < best_val:
            save_checkpoint(last_ckpt_path, model, optimizer, scheduler, epoch, args)
            if avg_val < best_val:
                best_val = avg_val
                save_checkpoint(os.path.join(save_dir, "best_model.pt"), model, optimizer, scheduler, epoch, args,
                                best=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--project_name', type=str, default="tsp_FM_l40s")
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='spectral_trans')
    parser.add_argument('--save_dir', type=str, default="./checkpoints")
    parser.add_argument('--interpolant', type=str, default='linear')
    parser.add_argument('--stochasticity_scale', type=float, default=0.1)

    # Model Params
    parser.add_argument('--num_points', type=int, default=1000)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)

    # L40S DEFAULTS
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help="Virtual epoch length")

    # Training Params
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eta_min_factor', type=float, default=1e-3)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    train(args)