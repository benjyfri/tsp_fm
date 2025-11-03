import torch
import torch.optim as optim
import numpy as np
from torchcfm import ConditionalFlowMatcher
from tqdm import tqdm
import os
import argparse
import time

from tsp_flow.data_loader import get_loaders
from tsp_flow.models import StrongEquivariantVectorField
from tsp_flow.utils import count_parameters, plot_loss_curves


def main(args):
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    print("Loading data...")
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
    cfm = ConditionalFlowMatcher(sigma=0.0)

    # Training
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')

    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False)

        for x0, x1, _ in train_pbar:
            x0, x1 = x0.to(device), x1.to(device)
            optimizer.zero_grad()

            t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
            vt = model(xt, t)
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
            for x0, x1 in test_pbar:
                x0, x1 = x0.to(device), x1.to(device)
                t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
                vt = model(xt, t)
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
    plot_loss_curves(train_losses, test_losses, os.path.join(args.output_dir, 'loss_curve.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Flow Matching for TSP")

    # Data Args
    parser.add_argument('--data_path', type=str, default="data_old_scripts/processed_tsp_dataset.pt", help='Path to the processed_tsp_dataset.pt file')
    parser.add_argument('--train_size', type=int, default=15000, help='Number of samples for the training set')

    # Model Args
    parser.add_argument('--num_points', type=int, default=15, help='Number of points in each cloud (N)')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension of the transformer')
    parser.add_argument('--t_emb_dim', type=int, default=128, help='Dimension of the time embedding')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')

    # Training Args
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # I/O Args
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save models and plots')

    args = parser.parse_args()
    main(args)