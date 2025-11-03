import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from tsp_flow.data_loader import get_loaders
from tsp_flow.models import StrongEquivariantVectorField
from tsp_flow.utils import save_animation, save_grid_plot


def main(args):
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    print("Loading data...")
    _, _, test_dataset = get_loaders(
        args.data_path,
        batch_size=args.num_samples,  # Batch size for grid plot
        train_split_size=args.train_size,
        seed=args.seed
    )
    if test_dataset is None:
        return

    # Model
    print("Loading model...")
    model = StrongEquivariantVectorField(
        n_points=args.num_points,
        embed_dim=args.embed_dim,
        t_emb_dim=args.t_emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    ).to(device)

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        return
    except Exception as e:
        print(f"Error loading model state: {e}")
        return

    model.eval()

    # --- Generate Animation ---
    anim_save_path = os.path.join(args.output_dir, f"tsp_flow_animation_sample_{args.anim_sample_idx}.gif")
    save_animation(
        model=model,
        test_dataset=test_dataset,
        device=device,
        save_path=anim_save_path,
        item_idx=args.anim_sample_idx
    )

    # --- Generate Grid Plot ---
    grid_save_path = os.path.join(args.output_dir, "tsp_flow_grid_plot.png")
    save_grid_plot(
        model=model,
        test_dataset=test_dataset,
        device=device,
        num_samples=args.num_samples,
        num_points=args.num_points,
        save_path=grid_save_path
    )

    print("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Flow Matching model for TSP")

    # Data Args
    parser.add_argument('--data_path', type=str, required=True, help='Path to the processed_tsp_dataset.pt file')
    parser.add_argument('--train_size', type=int, default=15000,
                        help='Number of samples for the training set (to ensure correct test split)')

    # Model Args
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model .pt file')
    parser.add_argument('--num_points', type=int, default=15, help='Number of points (must match trained model)')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension (must match trained model)')
    parser.add_argument('--t_emb_dim', type=int, default=128, help='Time embedding dim (must match trained model)')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers (must match trained model)')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads (must match trained model)')

    # Eval Args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples for the grid plot')
    parser.add_argument('--anim_sample_idx', type=int, default=0, help='Index of the test sample to animate')

    # I/O Args
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save plots and animations')

    args = parser.parse_args()
    main(args)