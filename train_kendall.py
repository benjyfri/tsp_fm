import torch
import torch.optim as optim
import numpy as np
# from torchcfm import ConditionalFlowMatcher # No longer needed for sampling
from tqdm import tqdm
import os
import argparse
import time

from tsp_flow.data_loader import get_loaders
from tsp_flow.models import StrongEquivariantVectorField
from tsp_flow.utils import count_parameters, plot_loss_curves


def sample_geodesic(x0, x1, theta, device, eps=1e-6):
    """
    [NEW] Samples a time t, location xt, and velocity ut along the
    Kendall's shape space geodesic.

    Args:
        x0 (torch.Tensor): Shape (B, N, 2), batch of start points.
        x1 (torch.Tensor): Shape (B, N, 2), batch of end points (aligned).
        theta (torch.Tensor): Shape (B,), batch of Procrustes distances.
        device (str): "cuda" or "cpu".
        eps (float): Small value for numerical stability.

    Returns:
        t (torch.Tensor): Shape (B,), sampled times.
        xt (torch.Tensor): Shape (B, N, 2), interpolated points.
        ut (torch.Tensor): Shape (B, N, 2), ground-truth velocity.
    """
    # Sample t and reshape for broadcasting
    t = torch.rand(x0.shape[0], device=device)
    t_ = t.view(-1, 1, 1)
    theta_ = theta.view(-1, 1, 1)

    # --- Geodesic Path (for theta > eps) ---
    a = (1 - t_) * theta_
    b = t_ * theta_
    sin_theta_ = torch.sin(theta_)

    # Add eps to denominator for stability
    xt_geo = (torch.sin(a) / (sin_theta_ + eps)) * x0 + (torch.sin(b) / (sin_theta_ + eps)) * x1
    ut_geo = (theta_ / (sin_theta_ + eps)) * (torch.cos(b) * x1 - torch.cos(a) * x0)

    # --- Linear Path (for theta <= eps) ---
    xt_lin = (1 - t_) * x0 + t_ * x1
    ut_lin = x1 - x0

    # --- Combine based on theta ---
    # Create a mask for broadcasting
    mask = (theta_ <= eps)

    xt = torch.where(mask, xt_lin, xt_geo)
    ut = torch.where(mask, ut_lin, ut_geo)

    return t, xt, ut


def main(args):
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    print("Loading data...")
    # [ASSUMPTION] get_loaders now loads the new .pt file and the loaders
    # yield batches of (x0, x1, theta).
    train_loader, test_loader, _ = get_loaders(
        args.data_path,
        args.batch_size,
        args.train_size,
        args.seed
    )
    if train_loader is None:
        return

    # Model
    print("Building model...")
    model = StrongEquivariantVectorField(
        n_points=args.num_points,
        embed_dim=args.embed_dim,
        t_emb_dim=args.t_emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    print(f"Model architecture:\n{model}")
    print(f"Total trainable parameters: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # cfm = ConditionalFlowMatcher(sigma=0.0) # No longer needed

    # Training
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    stability_eps = 1e-6 # Epsilon for safe division

    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False)

        # [MODIFIED] Assuming loader now yields (x0, x1, theta)
        for x0, x1, theta, *_ in train_pbar:
            x0, x1, theta = x0.to(device), x1.to(device), theta.to(device)
            optimizer.zero_grad()

            # [MODIFIED] Use new geodesic sampling
            t, xt, ut = sample_geodesic(x0, x1, theta, device, stability_eps)

            vt = model(xt, t) # Pass t as (B,)
            loss = torch.mean((vt - ut) ** 2)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation (Test) Phase ---
        model.eval()
        epoch_test_loss = 0.0
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Test]", leave=False)

        with torch.no_grad():
            # [MODIFIED] Assuming loader now yields (x0, x1, theta)
            for x0, x1, theta, *_ in test_pbar:
                x0, x1, theta = x0.to(device), x1.to(device), theta.to(device)

                # [MODIFIED] Use new geodesic sampling
                t, xt, ut = sample_geodesic(x0, x1, theta, device, stability_eps)

                vt = model(xt, t) # Pass t as (B,)
                loss = torch.mean((vt - ut) ** 2)
                epoch_test_loss += loss.item()
                test_pbar.set_postfix(loss=loss.item())

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  -> New best model saved with test loss: {best_test_loss:.6f}")

    end_time = time.time()
    print(f"Training complete in {(end_time - start_time) / 60:.2f} minutes.")

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
    print(f"Final model saved to {args.output_dir}/final_model.pt")

    # Plot and save loss curves
    plot_path = os.path.join(args.output_dir, 'loss_curve.png')
    plot_loss_curves(train_losses, test_losses, plot_path)
    print(f"Loss curve saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flow Matching for TSP (Kendall Geodesic)")

    # Data Args
    # [MODIFIED] Update default path to point to a new file
    parser.add_argument('--data_path', type=str, default="/home/benjamin.fri/PycharmProjects/tsp_fm/data_old_scripts/processed_tsp_dataset_TSP50_train.pt", help='Path to the processed .pt file (Kendall format)')
    parser.add_argument('--train_size', type=int, default=900000, help='Number of samples for the training set')

    # Model Args
    parser.add_argument('--num_points', type=int, default=50, help='Number of points in each cloud (N)')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension of the transformer')
    parser.add_argument('--t_emb_dim', type=int, default=128, help='Dimension of the time embedding')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')

    # Training Args
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # I/O Args
    parser.add_argument('--output_dir', type=str, default='checkpoints_kendall', help='Directory to save models and plots')

    args = parser.parse_args()
    main(args)