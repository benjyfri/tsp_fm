#!/usr/bin/env python3
"""
Fully revised training + diagnostics script for Flow Matching on Kendall/Pre-shape space.

Key fixes included:
- Force default dtype to float32.
- Robust geomstats CPU/GPU handling using CPU torch tensors (float32) when needed.
- Small nonzero init for final output layer to avoid zero-gradient stall.
- Cast geomstats outputs to model param dtype right before forward.
- Detailed diagnostics: vt/ut/xt dtypes, pre-final activation h stats, grad norms for first batch.
- Tiny overfit test uses lr=1e-2 and weight_decay=0 for quick debugging.

Run:
python demo_test_geomstats.py --train_data_path data_old_scripts/geom_demo_train_N100.pt \
    --test_data_path data_old_scripts/geom_demo_val_N10.pt --num_points 50 --run_tiny_test
"""

import os
import sys
import time
import argparse
from pathlib import Path

# --- MUST set geomstats backend before importing geomstats ---
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
# ------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Force default dtype to float32 to avoid surprising float64 tensors.
torch.set_default_dtype(torch.float32)

import geomstats.backend as gs
from geomstats.geometry.pre_shape import PreShapeSpace

# plotting (non-interactive)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# try to seed geomstats RNG
try:
    gs.random.seed(42)
except Exception:
    pass


# ----------------------------- Utilities ------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def device_info(device):
    print(f"Using device: {device}")
    if 'cuda' in str(device):
        print(" CUDA available?", torch.cuda.is_available())
        try:
            print(" GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        except Exception:
            pass


# Pure-torch projection to tangent space of pre-shape sphere (differentiable)
# v_raw, x: (B, N, D)
def project_to_tangent_torch(v_raw, x, eps=1e-9):
    inner = (v_raw * x).sum(dim=2, keepdim=True)
    denom = (x * x).sum(dim=2, keepdim=True).clamp(min=eps)
    v_t = v_raw - (inner / denom) * x
    return v_t


# Ensure conversion of geomstats outputs to torch on correct device/dtype
def ensure_torch(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    try:
        arr = gs.to_numpy(x)
    except Exception:
        arr = np.asarray(x)
    return torch.as_tensor(arr, device=device, dtype=dtype)


def debug_tensor_info(name, t):
    if isinstance(t, torch.Tensor):
        try:
            print(f"{name}: torch device={t.device}, dtype={t.dtype}, shape={tuple(t.shape)}, mean={t.mean().item():.6e}, std={t.std().item():.6e}")
        except Exception:
            print(f"{name}: torch tensor, device={t.device}, dtype={t.dtype}, shape={tuple(t.shape)}")
    elif isinstance(t, np.ndarray):
        print(f"{name}: numpy array shape={t.shape}, mean={t.mean():.6e}, std={t.std():.6e}")
    else:
        print(f"{name}: type={type(t)}")


# ----------------------------- Model -----------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.max_period = 10000

    def forward(self, t):
        B = t.shape[0]
        half = max(1, self.dim // 2)
        device = t.device
        log_max = torch.log(torch.tensor(self.max_period, device=device, dtype=torch.float32))
        denom = torch.tensor(max(1, half - 1), device=device, dtype=torch.float32)
        freqs = torch.exp(torch.arange(0, half, device=device, dtype=torch.float32) * (-log_max / denom))
        args = t.view(B, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            pad = torch.zeros(B, self.dim - emb.shape[1], device=device)
            emb = torch.cat([emb, pad], dim=1)
        return emb


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        return x


class KendallVectorFieldModel(nn.Module):
    def __init__(self, n_points=50, embed_dim=256, t_emb_dim=128, num_layers=4, num_heads=8, dropout=0.0):
        super().__init__()
        self.n_points = n_points
        self.input_dim = 2
        self.embed_dim = embed_dim
        self.time_embed = TimeEmbedding(t_emb_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim + t_emb_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.input_dim)
        )
        # small non-zero init for final layer so gradients flow initially
        nn.init.xavier_uniform_(self.output_head[-1].weight, gain=0.001)
        nn.init.zeros_(self.output_head[-1].bias)

        # debug flag to print pre-final activation stats once
        self._debug_print_h = False

    def forward(self, x, t, space=None):
        B, N, D = x.shape
        t_emb = self.time_embed(t)
        t_exp = t_emb.unsqueeze(1).repeat(1, N, 1)
        inp = torch.cat([x, t_exp], dim=2)
        h = self.input_projection(inp)
        for block in self.transformer_blocks:
            h = block(h)

        # debug pre-output activation if enabled
        if getattr(self, '_debug_print_h', False):
            try:
                print(f"DEBUG h stats: mean={h.mean().item():.6e}, std={h.std().item():.6e}, dtype={h.dtype}, device={h.device}")
            except Exception:
                pass

        v_raw = self.output_head(h)
        v_tangent = project_to_tangent_torch(v_raw, x)
        return v_tangent


# ----------------------------- Data Loading ----------------------------
def load_dataset(data_path, device):
    print(f"Loading data from: {data_path}")
    try:
        data = torch.load(data_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    if not isinstance(data, list) or len(data) == 0:
        print("Error: Dataset is empty or invalid")
        sys.exit(1)
    x0_list, x1_list, theta_list = [], [], []
    for entry in data:
        x0_list.append(torch.from_numpy(entry['points']).float())
        x1_list.append(torch.from_numpy(entry['circle']).float())
        theta_list.append(float(entry.get('theta', 0.0)))
    x0 = torch.stack(x0_list).to(device)
    x1 = torch.stack(x1_list).to(device)
    theta = torch.tensor(theta_list, dtype=torch.float32, device=device)
    print(f"Loaded {x0.shape[0]} samples")
    return x0, x1, theta


def create_batches(x0, x1, theta, batch_size, shuffle=True):
    n_samples = x0.shape[0]
    indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield {'x0': x0[batch_indices], 'x1': x1[batch_indices], 'theta': theta[batch_indices]}


# ----------------------------- Geomstats Space -------------------------
def setup_geomstats_space(n_points, ambient_dim=2):
    space = PreShapeSpace(k_landmarks=n_points, ambient_dim=ambient_dim, equip=True)
    metric = space.metric
    print(f"Created PreShapeSpace:")
    print(f"  - Dimension: {space.dim}")
    print(f"  - Number of landmarks: {space.k_landmarks}")
    print(f"  - Ambient dimension: {space.ambient_dim}")
    return space, metric


def sample_geodesic_geomstats(x0, x1, metric, device):
    """
    Robust geodesic sampling that avoids mixed-device and dtype errors.

    Strategy:
    - Try to set geomstats default device if available (preferred).
    - Otherwise call geomstats functions on CPU torch tensors with float32 dtype
      and move results to `device`.
    """
    B = x0.shape[0]
    canonical_dtype = x0.dtype if isinstance(x0, torch.Tensor) else torch.float32
    t = torch.rand(B, device=device, dtype=canonical_dtype)

    def _call_geom_on_cpu(fn, *args):
        cpu_args = []
        for a in args:
            if isinstance(a, torch.Tensor):
                cpu_args.append(a.detach().cpu().to(dtype=torch.float32))
            else:
                cpu_args.append(a)
        cpu_res = fn(*cpu_args)
        if isinstance(cpu_res, torch.Tensor):
            return cpu_res.to(device=device, dtype=canonical_dtype)
        try:
            arr = gs.to_numpy(cpu_res)
        except Exception:
            arr = np.asarray(cpu_res)
        return torch.as_tensor(arr, device=device, dtype=canonical_dtype)

    tried_set = False
    try:
        if hasattr(gs, 'set_default_device'):
            gs.set_default_device(str(device))
            tried_set = True
        if hasattr(gs, 'set_default_dtype'):
            try:
                gs.set_default_dtype(np.float32)
            except Exception:
                pass
    except Exception:
        tried_set = False

    if tried_set:
        log_raw = metric.log(x1, x0)
        log = ensure_torch(log_raw, device, dtype=canonical_dtype)
    else:
        log = _call_geom_on_cpu(metric.log, x1, x0)

    t_b = t.view(B, 1, 1)
    geodesic_segment = t_b * log

    if tried_set:
        xt_raw = metric.exp(geodesic_segment, x0)
        xt = ensure_torch(xt_raw, device, dtype=canonical_dtype)
    else:
        xt = _call_geom_on_cpu(metric.exp, geodesic_segment, x0)

    # Use projection of log at xt as transported velocity surrogate (tangent projection)
    ut = project_to_tangent_torch(log, xt)

    # Final safety cast: ensure everything is same dtype as x0 and on target device
    if not isinstance(xt, torch.Tensor):
        xt = torch.as_tensor(xt, device=device, dtype=canonical_dtype)
    else:
        xt = xt.to(device=device, dtype=canonical_dtype)
    if not isinstance(log, torch.Tensor):
        log = torch.as_tensor(log, device=device, dtype=canonical_dtype)
    else:
        log = log.to(device=device, dtype=canonical_dtype)
    ut = ut.to(device=device, dtype=canonical_dtype)

    # Sanity check
    if not (xt.device == device and ut.device == device and log.device == device):
        raise RuntimeError(f"Devices mismatch: xt {xt.device}, ut {ut.device}, log {log.device}, expected {device}")
    if not (xt.dtype == canonical_dtype and ut.dtype == canonical_dtype and log.dtype == canonical_dtype):
        raise RuntimeError(f"Dtype mismatch: xt {xt.dtype}, ut {ut.dtype}, log {log.dtype}, expected {canonical_dtype}")

    return t, xt, ut


# ----------------------------- Training Utilities ----------------------
def check_preshape(x, name="x"):
    means = x.mean(dim=1)
    max_mean_abs = means.abs().max().item()
    frob = x.view(x.shape[0], -1).norm(dim=1)
    min_norm = frob.min().item(); max_norm = frob.max().item()
    print(f"CHECK preshape {name}: max |mean|={max_mean_abs:.3e}, norm min/max={min_norm:.6e}/{max_norm:.6e}")
    return max_mean_abs, min_norm, max_norm


def to_preshape(x, eps=1e-12):
    x = x - x.mean(dim=1, keepdim=True)
    norms = x.view(x.shape[0], -1).norm(dim=1, keepdim=True)
    norms = norms.clamp(min=eps)
    x = x / norms.view(-1, 1, 1)
    return x


def debug_grad_and_types(model, vt, ut):
    print("--- DEBUG GRAD/TYPE ---")
    debug_tensor_info('vt', vt)
    debug_tensor_info('ut', ut)
    for i, (name, p) in enumerate(model.named_parameters()):
        if p.requires_grad:
            g = p.grad
            print(f"param {name}: grad is None? {g is None}", end='')
            if g is not None:
                try:
                    print(f"; norm={g.norm().item():.6e}; mean={g.mean().item():.6e}")
                except Exception:
                    print()
            else:
                print()
        if i >= 4:
            break
    print("------------------------")


# ----------------------------- Training Loop --------------------------
def train_epoch(model, x0, x1, theta, space, metric, optimizer, args, device, debug_first_batch=False):
    model.train()
    total_loss = 0.0
    n_batches = 0

    # canonical dtype from model params
    param_dtype = next(model.parameters()).dtype

    for batch in create_batches(x0, x1, theta, args.batch_size, shuffle=True):
        bx0 = batch['x0'].to(device)
        bx1 = batch['x1'].to(device)

        t, xt, ut = sample_geodesic_geomstats(bx0, bx1, metric, device)

        # Force-cast geoms to model dtype (prevents Float/Double mismatches)
        if xt.dtype != param_dtype:
            xt = xt.to(dtype=param_dtype, device=device)
        if t.dtype != param_dtype:
            t = t.to(dtype=param_dtype, device=device)
        if ut.dtype != param_dtype:
            ut = ut.to(dtype=param_dtype, device=device)

        # debug prints AFTER casting
        if debug_first_batch and n_batches == 0:
            debug_tensor_info('t (after cast)', t)
            debug_tensor_info('xt (after cast)', xt)
            debug_tensor_info('ut (after cast)', ut)

        optimizer.zero_grad()
        vt = model(xt, t, space)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()

        if debug_first_batch and n_batches == 0:
            debug_grad_and_types(model, vt, ut)

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def validate(model, x0, x1, theta, space, metric, args, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    param_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for batch in create_batches(x0, x1, theta, args.batch_size, shuffle=False):
            bx0 = batch['x0'].to(device)
            bx1 = batch['x1'].to(device)
            t, xt, ut = sample_geodesic_geomstats(bx0, bx1, metric, device)

            # cast to model dtype
            if xt.dtype != param_dtype:
                xt = xt.to(dtype=param_dtype, device=device)
            if t.dtype != param_dtype:
                t = t.to(dtype=param_dtype, device=device)
            if ut.dtype != param_dtype:
                ut = ut.to(dtype=param_dtype, device=device)

            vt = model(xt, t, space)
            loss = torch.mean((vt - ut) ** 2)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


def tiny_overfit_test(train_x0, train_x1, metric, space, device):
    print("--- Tiny overfit test (1 sample) ---")
    s0 = train_x0[:1].clone()
    s1 = train_x1[:1].clone()

    tiny = KendallVectorFieldModel(n_points=s0.shape[1], embed_dim=64, t_emb_dim=16, num_layers=1, num_heads=1).to(device).float()
    opt = torch.optim.AdamW(tiny.parameters(), lr=1e-2, weight_decay=0.0)
    param_dtype = next(tiny.parameters()).dtype

    for i in range(400):
        t, xt, ut = sample_geodesic_geomstats(s0, s1, metric, device)

        # cast to tiny model dtype
        if xt.dtype != param_dtype:
            xt = xt.to(dtype=param_dtype, device=device)
        if t.dtype != param_dtype:
            t = t.to(dtype=param_dtype, device=device)
        if ut.dtype != param_dtype:
            ut = ut.to(dtype=param_dtype, device=device)

        opt.zero_grad()
        vt = tiny(xt, t, space)
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        opt.step()
        if i % 50 == 0:
            print(f"tiny step {i}: loss={loss.item():.6e}")
    print("tiny overfit done")


def save_loss_plot(train_losses, test_losses, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='train')
    plt.plot(range(1, len(test_losses) + 1), test_losses, 'r-', label='test')
    plt.xlabel('epoch')
    plt.ylabel('mse loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved loss plot to {output_path}")


# ----------------------------- Main -----------------------------------
def main(args):
    set_seed(args.seed)

    # Device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    device_info(device)

    # Try to instruct geomstats to allocate on same device if available
    tried_set = False
    try:
        if hasattr(gs, 'set_default_device'):
            gs.set_default_device(str(device))
            tried_set = True
            print(f"geomstats.set_default_device -> {device}")
    except Exception:
        print("geomstats.set_default_device not available or failed; falling back to CPU-call wrapper for geom ops")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Setup space and metric
    space, metric = setup_geomstats_space(args.num_points, ambient_dim=2)

    # Load data
    train_x0, train_x1, train_theta = load_dataset(args.train_data_path, device)
    test_x0, test_x1, test_theta = load_dataset(args.test_data_path, device)

    print(f"Train samples: {train_x0.shape[0]}")
    print(f"Test samples: {test_x0.shape[0]}")

    # Verify preshape
    print("Verifying data preprocessing...")
    for i in range(min(5, train_x0.shape[0])):
        x_np = train_x0[i].cpu().numpy()
        mean = np.mean(x_np, axis=0)
        print(f"  Sample {i}: mean = {mean}, norm = {np.linalg.norm(x_np, 'fro'):.6f}")

    # Optional convert to preshape if needed
    max_mean, min_norm, max_norm = check_preshape(train_x0, 'train_x0')
    if max_mean > 1e-6 or abs(min_norm - 1.0) > 1e-3:
        print("Converting to preshape (centering+normalization)")
        train_x0 = to_preshape(train_x0)
        train_x1 = to_preshape(train_x1)
        test_x0 = to_preshape(test_x0)
        test_x1 = to_preshape(test_x1)

    # Model
    model = KendallVectorFieldModel(n_points=args.num_points, embed_dim=args.embed_dim, t_emb_dim=args.t_emb_dim,
                                    num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout).to(device)
    model = model.float()
    # print a param dtype check
    first_name, first_param = next(model.named_parameters())
    print(f"first param: {first_name}, dtype={first_param.dtype}, device={first_param.device}")
    # enable pre-final activation debug for the first epoch
    model._debug_print_h = True

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # Tiny test option
    if args.run_tiny_test:
        tiny_overfit_test(train_x0, train_x1, metric, space, device)

    # One debug epoch first
    print("--- Running one debug epoch (first-batch prints) ---")
    _ = train_epoch(model, train_x0, train_x1, train_theta, space, metric, optimizer, args, device, debug_first_batch=True)

    # disable h debug after first debug epoch
    model._debug_print_h = False

    train_losses = []
    test_losses = []
    best_test = float('inf')

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        start = time.time()
        tr = train_epoch(model, train_x0, train_x1, train_theta, space, metric, optimizer, args, device)
        te = validate(model, test_x0, test_x1, test_theta, space, metric, args, device)
        train_losses.append(tr)
        test_losses.append(te)
        scheduler.step()
        elapsed = time.time() - start
        print(f"Epoch {epoch + 1}/{args.epochs}: train={tr:.6e}, test={te:.6e}, lr={optimizer.param_groups[0]['lr']:.2e}, time={elapsed:.2f}s")
        if te < best_test:
            best_test = te
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, out / 'best_model.pt')

    # Save final artifacts (outside epoch loop)
    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict(), 'train_losses': train_losses, 'test_losses': test_losses}, out / 'final_model.pt')
    save_loss_plot(train_losses, test_losses, out / 'loss_plot.png')
    print("Training complete. Outputs in:", out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train flow matching model for Kendall shape space")
    parser.add_argument('--train_data_path', type=str, default="data_old_scripts/geom_demo_train_N100.pt")
    parser.add_argument('--test_data_path', type=str, default="data_old_scripts/geom_demo_val_N10.pt")
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--grad_clip_norm', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='checkpoints_kendall_geomstats')
    parser.add_argument('--run_tiny_test', action='store_true')
    args = parser.parse_args()
    main(args)
