import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from torch.utils.data import DataLoader

from tsp_flow.data_loader import get_loaders
from tsp_flow.models import StrongEquivariantVectorField
from tsp_flow.utils import save_animation, save_grid_plot


# --- NEW HELPER FUNCTIONS ---

def calculate_tsp_length(points, order):
    """
    Calculates the total length of a TSP tour given points and an order.
    'points' is (N, 2)
    'order' is (N,)
    """
    # Reorder the points according to the predicted tour
    ordered_points = points[order]

    # Calculate Euclidean distances between consecutive points in the tour
    # torch.roll(..., shifts=1, ...) creates the (p[i], p[i-1]) pairs
    distances = torch.norm(ordered_points - torch.roll(ordered_points, shifts=1, dims=0), dim=1)

    # Sum the distances to get the total tour length
    total_length = torch.sum(distances)
    return total_length

def get_angular_order(points):
    """
    Finds the node order by sorting points angularly around their centroid.
    'points' is (N, 2), representing the predicted circle.
    """
    # Calculate the centroid (mean) of the points
    centroid = torch.mean(points, dim=0, keepdim=True)

    # Center the points by subtracting the centroid
    centered_points = points - centroid

    # Calculate the angle of each point using atan2
    angles = torch.atan2(centered_points[:, 1], centered_points[:, 0])

    # Get the indices that would sort the points by angle
    order = torch.argsort(angles)
    return order

@torch.no_grad()
def run_inference_batch(model, x0_batch, device, ode_steps):
    """
    Runs the flow matching inference on a batch of samples from t=0 to t=1.
    'x0_batch' is (B, N, 2)
    """

    # The function for odeint must have the signature func(t, y)
    # where y is the state (B, N, 2) and t is a scalar time
    def ode_func(t, y):
        # Expand t to a batch dimension: (B,)
        t_batch = t.expand(y.shape[0]).to(device)
        # Get the vector field from the model
        v = model(y, t_batch)
        return v

    # Define the time span for integration
    t_span = torch.linspace(0., 1., steps=ode_steps).to(device)

    # Run the ODE solver
    # traj will have shape (ode_steps, B, N, 2)
    traj = odeint(
        ode_func,
        x0_batch.to(device),
        t_span,
        method='dopri5'  # A good default adaptive solver
    )

    # The final predicted points are at the last time step
    x1_pred_batch = traj[-1]
    return x1_pred_batch

# --- END OF HELPER FUNCTIONS ---


def main(args):
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    print("Loading data...")
    # This assumes your get_loaders returns a test_dataset object
    # that implements __getitem__ to return (x0, x1, total_length)
    _, _, test_dataset = get_loaders(
        args.data_path,
        batch_size=args.num_samples,  # This batch size is for the grid plot
        train_split_size=args.train_size,
        seed=args.seed
    )
    if test_dataset is None:
        return

    # Check if we have enough samples for animation
    if args.num_animations > len(test_dataset):
        print(f"Warning: Requested {args.num_animations} animations, but test set only has {len(test_dataset)} samples.")
        print(f"Will generate {len(test_dataset)} animations instead.")
        args.num_animations = len(test_dataset)

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

    # --- NEW: Calculate Statistics ---
    print("Calculating statistics over the entire test set...")
    # Use a DataLoader for efficient batch processing
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    all_percentage_errors = []

    for batch_idx, batch in enumerate(test_loader):
        # Assumes dataset returns (x0, x1, total_length)
        x0_batch, _, gt_length_batch = batch

        x0_batch = x0_batch.to(device)
        gt_length_batch = gt_length_batch.to(device)

        # Run batched inference to get the predicted circles
        x1_pred_batch = run_inference_batch(model, x0_batch, device, args.ode_steps)

        # Process each sample in the batch
        for i in range(x0_batch.shape[0]):
            x0_sample = x0_batch[i]
            gt_length_sample = gt_length_batch[i]
            x1_pred_sample = x1_pred_batch[i]

            # 1. Get Hamiltonian cycle from model output (predicted circle)
            pred_order = get_angular_order(x1_pred_sample)

            # 2. Calculate tour length on original graph (x0)
            pred_length = calculate_tsp_length(x0_sample, pred_order)

            # 3. Calculate percentage error
            error = (pred_length - gt_length_sample) / gt_length_sample
            all_percentage_errors.append(error.item())

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(test_loader):
            print(f"  Processed { (batch_idx + 1) * args.batch_size } / { len(test_dataset) } samples...")

    # Compute and print statistics
    errors_tensor = torch.tensor(all_percentage_errors)
    mean_err = errors_tensor.mean()
    std_err = errors_tensor.std()
    min_err = errors_tensor.min()
    max_err = errors_tensor.max()

    print("\n" + "="*30)
    print("--- TSP Tour Length Evaluation ---")
    print(f"Evaluated on {len(errors_tensor)} test samples.")
    print(f"Percentage Error ( (Pred_Length - GT_Length) / GT_Length ):")
    print(f"  Mean:    {mean_err.item():.4f} (or {mean_err.item()*100:.2f}%)")
    print(f"  Std Dev: {std_err.item():.4f} (or {std_err.item()*100:.2f}%)")
    print(f"  Min:     {min_err.item():.4f} (or {min_err.item()*100:.2f}%)")
    print(f"  Max:     {max_err.item():.4f} (or {max_err.item()*100:.2f}%)")
    print("="*30 + "\n")

    # --- Generate 5 Animations ---
    print(f"Generating {args.num_animations} animations...")
    for i in range(args.num_animations):
        print(f"  Creating animation for sample {i}...")
        anim_save_path = os.path.join(args.output_dir, f"tsp_flow_animation_sample_{i}.gif")
        save_animation(
            model=model,
            test_dataset=test_dataset,
            device=device,
            save_path=anim_save_path,
            item_idx=i  # Use the loop index for each sample
        )
    print(f"Finished generating {args.num_animations} animations.")

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
    parser.add_argument('--data_path', type=str, default="data_old_scripts/processed_tsp_dataset.pt", help='Path to the processed_tsp_dataset.pt file')
    parser.add_argument('--train_size', type=int, default=15000,
                        help='Number of samples for the training set (to ensure correct test split)')

    # Model Args
    parser.add_argument('--model_path', type=str, default="checkpoints/final_model.pt", help='Path to the saved model .pt file')
    parser.add_argument('--num_points', type=int, default=15, help='Number of points (must match trained model)')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension (must match trained model)')
    parser.add_argument('--t_emb_dim', type=int, default=128, help='Time embedding dim (must match trained model)')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers (must match trained model)')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads (must match trained model)')

    # Eval Args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples for the grid plot')
    parser.add_argument('--num_animations', type=int, default=5, help='Number of separate animations to generate')

    # --- NEW ARGS ---
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for statistics evaluation')
    parser.add_argument('--ode_steps', type=int, default=100, help='Number of steps for ODE solver')

    # I/O Args
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save plots and animations')

    args = parser.parse_args()
    main(args)