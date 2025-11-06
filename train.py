import torch
import torch.optim as optim
import numpy as np
from torchcfm import ConditionalFlowMatcher
from tqdm import tqdm
import os
import argparse
import time
# [NEW] Import for LR Scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tsp_flow.data_loader import get_loaders
from tsp_flow.models import StrongEquivariantVectorField
from tsp_flow.utils import count_parameters, plot_loss_curves

# --- Helper Function for Kendall Geodesic Sampling ---
def sample_geodesic(x0, x1, theta, device, eps=1e-6):
    """
    Samples a time t, location xt, and velocity ut along the
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

# --- [NEW] Generic Flow Sampling Function ---
def get_flow_sample(batch, method, device, cfm_obj=None, geo_eps=1e-6):
    """
    Generic function to sample a flow (t, xt, ut) based on the chosen method.

    Args:
        batch: The batch of data from the data loader.
        method (str): The flow matching method ('linear' or 'kendall').
        device (str): "cuda" or "cpu".
        cfm_obj (ConditionalFlowMatcher, optional): CFM object for 'linear' method.
        geo_eps (float, optional): Epsilon for 'kendall' method stability.

    Returns:
        t (torch.Tensor): Shape (B,), sampled times.
        xt (torch.Tensor): Shape (B, N, 2), interpolated points.
        ut (torch.Tensor): Shape (B, N, 2), ground-truth velocity.
    """
    # Common data unpacking
    x0, x1 = batch[0].to(device), batch[1].to(device)

    if method == 'linear':
        if cfm_obj is None:
            raise ValueError("ConditionalFlowMatcher object must be provided for 'linear' method.")
        t, xt, ut = cfm_obj.sample_location_and_conditional_flow(x0, x1)

    elif method == 'kendall':
        if len(batch) < 3:
            # Ensure data loader is providing theta
            raise ValueError("Kendall method requires theta (as 3rd batch element). Data loader is not providing it.")
        theta = batch[2].to(device)
        t, xt, ut = sample_geodesic(x0, x1, theta, device, geo_eps)

    # [Future] Add more methods here
    # elif method == 'my_other_method':
    #     t, xt, ut = ...

    else:
        raise NotImplementedError(f"FM Method '{method}' not implemented.")

    return t, xt, ut


def main(args):
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"  -> GPU Name: {torch.cuda.get_device_name(0)}")

    # Handle output directory
    if args.output_dir is None:
        args.output_dir = f"checkpoints_{args.fm_method}"

    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using Flow Matching method: {args.fm_method}")

    # Data
    print("Loading data...")
    train_loader, test_loader, _ = get_loaders(
        args.train_data_path,
        args.test_data_path,
        args.batch_size
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

    # [NEW] Add Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        'min',            # Mode: look at the 'min' value of the loss
        factor=args.lr_factor, # Factor to reduce LR by (e.g., 0.5)
        patience=args.lr_patience # Wait this many epochs with no improvement
    )
    print(f"Using ReduceLROnPlateau LR scheduler with patience {args.lr_patience} and factor {args.lr_factor}.")


    # Conditional CFM initialization
    cfm = None
    stability_eps = 1e-6 # Only used by 'kendall'
    if args.fm_method == 'linear':
        print("Initializing ConditionalFlowMatcher for Linear FM.")
        cfm = ConditionalFlowMatcher(sigma=0.0)
    elif args.fm_method == 'kendall':
        print("Using Kendall Geodesic sampling.")

    # Training
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')

    print(f"Starting training for {args.log_epochs} epochs...") # Changed from args.epochs
    start_time = time.time()

    for epoch in range(args.log_epochs): # Changed from args.epochs
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.log_epochs} [Train]", leave=False) # Changed from args.epochs

        for batch_idx, batch in enumerate(train_pbar):
            optimizer.zero_grad()

            # Use generic sampling function
            try:
                t, xt, ut = get_flow_sample(batch, args.fm_method, device, cfm_obj=cfm, geo_eps=stability_eps)
            except (ValueError, NotImplementedError) as e:
                print(f"Error during training: {e}")
                print("Exiting...")
                return # Exit on data/method error

            vt = model(xt, t)
            loss = torch.mean((vt - ut) ** 2)

            # [NEW] Diagnostic Logging for Kendall
            # Log 1st batch of every 5th epoch (or first/last epoch)
            if (args.fm_method == 'kendall' and
                    batch_idx == 0 and
                    (epoch % 5 == 0 or epoch == 0 or epoch == args.log_epochs - 1)):

                try:
                    ut_norm = torch.norm(ut, p=2, dim=(-1, -2)).mean().item()
                    vt_norm = torch.norm(vt.detach(), p=2, dim=(-1, -2)).mean().item()
                    theta_mean = torch.mean(batch[2]).item() # Assumes batch[2] is theta
                    theta_max = torch.max(batch[2]).item()

                    print(f"\n[DIAGNOSTIC (Epoch {epoch+1}, Batch 0)]")
                    print(f"  -> Theta (mean/max): {theta_mean:.4f} / {theta_max:.4f} (max near 3.14 is unstable)")
                    print(f"  -> GT Vel Norm (ut): {ut_norm:.6f}")
                    print(f"  -> Model Vel Norm (vt): {vt_norm:.6f}")
                    print(f"  -> Current LR: {optimizer.param_groups[0]['lr']:.2e}")
                except Exception as e:
                    print(f"\n[DIAGNOSTIC] Error during logging: {e}")


            loss.backward()

            # [NEW] Gradient Clipping
            # Clip gradients to prevent explosion, especially for 'kendall'
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)

            optimizer.step()

            epoch_train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation (Test) Phase ---
        model.eval()
        epoch_test_loss = 0.0
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{args.log_epochs} [Test]", leave=False) # Changed from args.epochs

        with torch.no_grad():
            for batch in test_pbar:

                # Use generic sampling function
                try:
                    t, xt, ut = get_flow_sample(batch, args.fm_method, device, cfm_obj=cfm, geo_eps=stability_eps)
                except (ValueError, NotImplementedError) as e:
                    print(f"Error during validation: {e}")
                    break # Exit inner loop

                vt = model(xt, t)
                loss = torch.mean((vt - ut) ** 2)
                epoch_test_loss += loss.item()
                test_pbar.set_postfix(loss=loss.item())

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch + 1}/{args.log_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}") # Changed from args.epochs

        # [NEW] Step the LR scheduler based on validation loss
        scheduler.step(avg_test_loss)

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
    parser = argparse.ArgumentParser(description="Train Flow Matching for TSP (Unified)")

    # --- Method Argument ---
    parser.add_argument('--fm_method', type=str, required=True,
                        choices=['linear', 'kendall'],
                        help='Flow matching method to use.')

    # Data Args
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Path to the training .pt file. Note: "kendall" method requires a file with (x0, x1, theta) tuples.')
    parser.add_argument('--test_data_path', type=str, required=True,
                        help='Path to the test/validation .pt file. Note: "kendall" method requires a file with (x0, x1, theta) tuples.')

    # Model Args
    parser.add_argument('--num_points', type=int, default=50,
                        help='Number of points in each cloud (N)')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Embedding dimension of the transformer')
    parser.add_argument('--t_emb_dim', type=int, default=128,
                        help='Dimension of the time embedding')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')

    # Training Args
    # [RENAMED] from --epochs to --log_epochs
    parser.add_argument('--log_epochs', type=int, default=100,
                        help='Number of training epochs. (Default: 100)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate (Default: 1e-4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # [NEW] Args for stabilization
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='Max norm for gradient clipping. (Set to 0 to disable). (Default: 1.0)')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='LR Scheduler: Epochs to wait for improvement. (Default: 10)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='LR Scheduler: Factor to reduce LR by. (Default: 0.5)')


    # I/O Args
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save models and plots. (default: checkpoints_[fm_method])')

    args = parser.parse_args()

    # [MODIFIED] Rename epochs to log_epochs in argparse namespace
    # to avoid conflict with the variable 'epochs' used in loops.
    # This just makes it clearer.
    # main(args)

    # Corrected: No need to rename, just use args.log_epochs consistently
    main(args)