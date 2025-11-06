import torch
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import matplotlib.animation as animation
import os


def count_parameters(model):
    """Returns the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_loss_curves(train_losses, test_losses, save_path):
    """Saves a plot of training and test loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title("Training and Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def generate_plot_data(model, x0_batch, x1_batch, t_span, device):
    """
    Helper function to run ODE solver for a batch of test samples.
    """
    model.eval()
    x0_batch = x0_batch.to(device)

    def ode_func(t, x):
        t_batch = torch.ones(x.shape[0]).to(device) * t
        with torch.no_grad():
            return model(x, t_batch)

    trajectory = odeint(ode_func, x0_batch, t_span, atol=1e-4, rtol=1e-4)

    pred_x1 = trajectory[-1].cpu().detach().numpy()
    x0_np = x0_batch.cpu().detach().numpy()
    x1_np = x1_batch.cpu().detach().numpy()

    return x0_np, x1_np, pred_x1


def save_animation(model, test_dataset, device, save_path, item_idx=0):
    """
    Generates and saves a GIF animation of the flow matching process for a single test sample.
    """
    print("Running ODE solver for animation...")
    model.eval()

    # [FIXED] Unpack 4 items from the dataset (x0, x1, theta, gt_length)
    x0_sample, x1_sample, _, _ = test_dataset[item_idx]

    x0_sample = x0_sample.unsqueeze(0).to(device)
    x1_sample = x1_sample.unsqueeze(0).to(device)

    def ode_func(t, x):
        t_batch = torch.ones(x.shape[0]).to(device) * t
        with torch.no_grad():
            return model(x, t_batch)

    t_span = torch.linspace(0, 1, 100).to(device)
    trajectory = odeint(ode_func, x0_sample, t_span, atol=1e-4, rtol=1e-4)
    print("Inference complete. Creating animation...")

    traj_np = trajectory.squeeze(1).cpu().detach().numpy()
    x0_np = x0_sample.squeeze(0).cpu().detach().numpy()
    x1_np = x1_sample.squeeze(0).cpu().detach().numpy()
    t_np = t_span.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    all_points = np.concatenate([x0_np, x1_np], axis=0)
    min_val, max_val = all_points.min() - 0.5, all_points.max() + 0.5
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.plot(x1_np[:, 0], x1_np[:, 1], 'gx', label='Target (x1)', markersize=10, mew=2)
    ax.plot(x0_np[:, 0], x0_np[:, 1], 'ro', label='Start (x0)', fillstyle='none', markersize=10, mew=2)

    points_plot = ax.plot(traj_np[0, :, 0], traj_np[0, :, 1], 'bo', label='Flowing (xt)')[0]
    ax.legend(loc='upper right')
    title = ax.set_title("Flow Matching (Test Sample): t=0.00")

    def update(frame):
        points_plot.set_data(traj_np[frame, :, 0], traj_np[frame, :, 1])
        title.set_text(f"Flow Matching (Test Sample): t={t_np[frame]:.2f}")
        return (points_plot, title)

    ani = animation.FuncAnimation(fig, update, frames=len(t_np), blit=True)

    # Save as GIF
    ani.save(save_path, writer='pillow', fps=10)
    plt.close()
    print(f"Animation saved to {save_path}")


def save_grid_plot(model, test_dataset, device, num_samples, num_points, save_path):
    """
    Generates and saves a grid plot comparing start, target, and predicted point clouds.
    """
    print(f"Running inference for {num_samples} grid samples...")
    # Create a temporary loader to get a batch
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_samples, shuffle=True)

    # [FIXED] Unpack 4 items from the loader
    x0_viz_batch, x1_viz_batch, _, _ = next(iter(loader))

    t_span_viz = torch.linspace(0, 1, 2).to(device)
    x0_plots, x1_plots, pred_x1_plots = generate_plot_data(model, x0_viz_batch, x1_viz_batch, t_span_viz, device=device)
    print("Inference complete. Plotting results...")

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:  # Make sure axes is always a 2D array
        axes = np.array([axes])

    fig.suptitle("Unconditional Flow Matching - Test Samples", fontsize=16, y=1.02)
    colors = plt.cm.jet(np.linspace(0, 1, num_points))

    for i in range(num_samples):
        x0, x1, pred = x0_plots[i], x1_plots[i], pred_x1_plots[i]
        all_coords = np.concatenate([x0, x1, pred], axis=0)
        min_val, max_val = all_coords.min() - 0.2, all_coords.max() + 0.2

        ax = axes[i, 0]
        ax.scatter(x0[:, 0], x0[:, 1], c=colors, s=50, edgecolors='k', alpha=0.7)
        ax.set_title(f"Sample {i + 1}: Start (x0)")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal')

        ax = axes[i, 1]
        ax.scatter(x1[:, 0], x1[:, 1], c=colors, s=50, edgecolors='k', alpha=0.7)
        ax.set_title(f"Sample {i + 1}: Ground Truth (x1)")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal')

        ax = axes[i, 2]
        ax.scatter(pred[:, 0], pred[:, 1], c=colors, s=50, edgecolors='k', alpha=0.7)
        ax.set_title(f"Sample {i + 1}: Prediction (pred_x1)")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Grid plot saved to {save_path}")