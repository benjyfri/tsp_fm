# Flow Matching for TSP

This project trains a Conditional Flow Matching (CFM) model to solve Traveling Salesperson Problems (TSP) by flowing a 2D point cloud to a target circle configuration.

## Project Structure

-   `train.py`: Main script to start model training.
-   `evaluate.py`: Script to run inference on a trained model and generate visualizations.
-   `tsp_flow/`: Python package containing the core logic.
    -   `data_loader.py`: Handles loading and processing the TSP dataset.
    -   `models.py`: Defines the `StrongEquivariantVectorField` transformer model.
    -   `utils.py`: Contains helper functions for plotting and parameter counting.
-   `requirements.txt`: Project dependencies.

## Setup

1.  Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

### Training

Place your `processed_tsp_dataset.pt` file in a known location.

```bash
python train.py --data_path /path/to/your/processed_tsp_dataset.pt --output_dir checkpoints