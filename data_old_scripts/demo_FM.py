import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torchcfm import ConditionalFlowMatcher
from torchdiffeq import odeint  # Use torchdiffeq for the ODE solver
from tqdm import tqdm  # Use standard tqdm
import matplotlib.animation as animation
import os
import sys

# --- 1. Setup ---

# Set a random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define dataset constants
NUM_POINTS = 15
# --- IMPORTANT ---
# Place 'processed_tsp_dataset.pt' in the same directory as this script.
DATA_FILE = 'processed_tsp_dataset.pt'


# --- 2. PyTorch Dataset ---

class PointCloudDataset(torch.utils.data.Dataset):
    """
    Loads the 'processed_tsp_dataset.pt' file.

    Each item returns the (noisy_input, tar_circle) pair.
    """

    def __init__(self, data_file):
        if not os.path.exists(data_file):
            print(f"Error: File not found at {data_file}.")
            print("Please make sure 'processed_tsp_dataset.pt' is in the same directory as this script.")
            sys.exit(1)

        try:
            # We use weights_only=False as the file contains pickled numpy arrays
            self.entries = torch.load(data_file, weights_only=False)
            print(f"Successfully loaded {len(self.entries)} total data entries.")
        except Exception as e:
            print(f"Error loading file: {e}")
            self.entries = []

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        x0 = torch.from_numpy(entry['points'].astype(np.float32))
        x1 = torch.from_numpy(entry['circle'].astype(np.float32))
        return x0, x1


# --- 3. Model (Vector Field) ---

class VectorField(nn.Module):
    """
    A simple MLP to model the vector field v(xt, t, x1).
    """

    def __init__(self, n_points=NUM_POINTS, hidden_dim=128):
        super().__init__()
        self.n_points = n_points
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, t, c):
        B, N, D = x.shape
        t_expanded = t.view(B, 1, 1).repeat(1, N, 1)
        inp = torch.cat([x, c, t_expanded], dim=2)
        return self.net(inp)


# --- Main execution block ---

def main():
    # --- Setup Device and DataLoaders ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        full_dataset = PointCloudDataset(DATA_FILE)

        if len(full_dataset) == 0:
            raise RuntimeError("Dataset is empty. Exiting.")

        TRAIN_SIZE = 2000
        TEST_SIZE = len(full_dataset) - TRAIN_SIZE

        if TEST_SIZE <= 0:
            raise ValueError(f"Full dataset size ({len(full_dataset)}) is not larger than TRAIN_SIZE ({TRAIN_SIZE}).")

        print(f"Splitting data into {TRAIN_SIZE} train and {TEST_SIZE} test samples.")

        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset,
            [TRAIN_SIZE, TEST_SIZE],
            generator=torch.Generator().manual_seed(42)
        )

        BATCH_SIZE = 128
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        x0_batch, x1_batch = next(iter(train_loader))
        print(f"\nBatch shapes: x0: {x0_batch.shape}, x1: {x1_batch.shape}")

    except Exception as e:
        print(e)
        return

    # --- Initialize Model and CFM ---
    model = VectorField(n_points=NUM_POINTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    cfm = ConditionalFlowMatcher(sigma=0.0)
    print("Model architecture:\n", model)

    # --- 5. Training Loop ---
    NUM_EPOCHS = 100
    train_losses = []
    test_losses = []
    print(f"Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]", leave=False)

        for x0, x1 in progress_bar:
            x0, x1 = x0.to(device), x1.to(device)
            optimizer.zero_grad()
            t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
            vt = model(xt, t, x1)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation (Test) Phase ---
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            progress_bar_test = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Test]", leave=False)
            for x0, x1 in progress_bar_test:
                x0, x1 = x0.to(device), x1.to(device)
                t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
                vt = model(xt, t, x1)
                loss = torch.mean((vt - ut) ** 2)
                epoch_test_loss += loss.item()
                progress_bar_test.set_postfix(loss=loss.item())

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Use tqdm.write to avoid interfering with progress bars
        tqdm.write(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

    print("Training complete!")

    # --- 6. Plot Training & Test Loss ---
    print("Generating loss plot...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title("Training and Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss.png")
    print("Saved training_loss.png")
    plt.show()

    # --- 7. Inference Animation (on Test Set) ---
    print("Generating inference animation...")
    model.eval()

    x0_sample, x1_sample = test_dataset[0]  # Get first test sample
    x0_sample = x0_sample.unsqueeze(0).to(device)
    x1_sample = x1_sample.unsqueeze(0).to(device)

    # Need to define ode_func within this scope to capture x1_sample
    def ode_func_anim(t, x):
        t_batch = torch.ones(x.shape[0]).to(device) * t
        with torch.no_grad():
            return model(x, t_batch, x1_sample)

    t_span = torch.linspace(0, 1, 100).to(device)
    trajectory = odeint(ode_func_anim, x0_sample, t_span, atol=1e-4, rtol=1e-4)

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

    # --- REVISED: Save animation as GIF ---
    try:
        ani.save('flow_animation.gif', writer='pillow', fps=15)
        print("Saved flow_animation.gif")
    except Exception as e:
        print(f"Could not save animation. Do you have 'pillow' installed? (pip install pillow)\nError: {e}")
    plt.close()

    # --- 8. Final Result Visualization (Prediction vs. GT) ---
    print("Generating static prediction vs. GT plot...")
    x_pred_np = traj_np[-1]
    final_mse = np.mean((x_pred_np - x1_np) ** 2)
    print(f"Final Mean Squared Error for this test sample: {final_mse:.6f}")

    plt.figure(figsize=(9, 9))
    plt.gca().set_aspect('equal')
    plt.plot(x1_np[:, 0], x1_np[:, 1], 'gx', label='Ground Truth (x1)', markersize=12, mew=3)
    plt.plot(x_pred_np[:, 0], x_pred_np[:, 1], 'bo', label='Model Prediction', markersize=10, alpha=0.7)
    plt.plot(x0_np[:, 0], x0_np[:, 1], 'r.', label='Start (x0)', markersize=4, alpha=0.5)

    for i in range(NUM_POINTS):
        plt.plot([x1_np[i, 0], x_pred_np[i, 0]], [x1_np[i, 1], x_pred_np[i, 1]], 'k--', alpha=0.6)

    plt.title(f"Final Prediction vs. Ground Truth (MSE: {final_mse:.6f})")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("prediction_vs_gt.png")
    print("Saved prediction_vs_gt.png")
    plt.show()

    # --- 9. 10-Sample Test Set Visualization ---
    print("Generating 10-sample test visualization...")

    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    axes = axes.flatten()
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_POINTS))

    for i in range(10):
        ax = axes[i]

        x0_sample, x1_sample = test_dataset[i]
        x0_sample = x0_sample.unsqueeze(0).to(device)
        x1_sample = x1_sample.unsqueeze(0).to(device)

        # Need to define ode_func in here to capture the correct x1_sample
        def ode_func_grid(t, x):
            t_batch = torch.ones(x.shape[0]).to(device) * t
            with torch.no_grad():
                return model(x, t_batch, x1_sample)

        t_span_grid = torch.linspace(0, 1, 20).to(device)
        trajectory_grid = odeint(ode_func_grid, x0_sample, t_span_grid, atol=1e-3, rtol=1e-3)

        x_pred_np_grid = trajectory_grid[-1].squeeze(0).cpu().detach().numpy()
        x1_np_grid = x1_sample.squeeze(0).cpu().detach().numpy()
        x0_np_grid = x0_sample.squeeze(0).cpu().detach().numpy()

        ax.set_aspect('equal')
        ax.scatter(x0_np_grid[:, 0], x0_np_grid[:, 1], c='gray', marker='.', alpha=0.3, label='Start (x0)')

        for k in range(NUM_POINTS):
            color = colors[k]
            ax.scatter(x1_np_grid[k, 0], x1_np_grid[k, 1], c=[color], marker='x', s=100, linewidth=2)
            ax.scatter(x_pred_np_grid[k, 0], x_pred_np_grid[k, 1], c=[color], marker='o', s=50, alpha=0.8)
            ax.plot([x1_np_grid[k, 0], x_pred_np_grid[k, 0]], [x1_np_grid[k, 1], x_pred_np_grid[k, 1]], c=color,
                    linestyle='--', alpha=0.7)

        if i == 0:
            ax.scatter([], [], c='black', marker='x', s=100, linewidth=2, label='Ground Truth (x1)')
            ax.scatter([], [], c='black', marker='o', s=50, alpha=0.8, label='Prediction')
            ax.legend(loc='best', fontsize='small')

        ax.set_title(f"Test Sample {i + 1}")
        ax.grid(True, linestyle='--', alpha=0.2)

    plt.suptitle("Test Set Predictions vs. Ground Truth (10 Samples)", fontsize=20, y=1.03)
    plt.tight_layout()
    plt.savefig("test_set_grid.png")
    print("Saved test_set_grid.png")
    plt.show()

    print("\nAll tasks complete. Plots and animation saved to disk.")


if __name__ == "__main__":
    main()