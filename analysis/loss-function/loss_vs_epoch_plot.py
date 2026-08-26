# Import necessary libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "text.usetex": True,  # Requires LaTeX installation
        "font.family": "serif",
        "font.size": 14,
        "axes.labelweight": "bold",
        "axes.titlesize": 16,
        "legend.fontsize": 14,
        "figure.titlesize": 16,
    }
)


def load_data(file_path, delimiter=",", skip_header=1):
    """
    Load data from a .dat file.

    Parameters:
        file_path (str): Path to the .dat file.
        delimiter (str): Delimiter used in the file.
        skip_header (int): Number of lines to skip at the beginning of the file.

    Returns:
        np.ndarray: Loaded data as a NumPy array.
    """
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    try:
        data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skip_header)
    except Exception as e:
        print(f"Error loading data from '{file_path}': {e}")
        sys.exit(1)

    return data


# Define the path to the data file

data_file = "loss_55-65_Hz_2048.dat"

# Load the data
data = load_data(data_file, delimiter=",", skip_header=1)

# Extract columns
# Step     Epochs    Train     Test
epochs = data[:, 1]  # Second column for epochs
train_loss = data[:, 2]  # Third column for train loss
test_loss = data[:, 3]  # Fourth column for test loss


# sns.set(style="whitegrid")  # Options: "darkgrid", "whitegrid", "ticks", etc.


plt.figure(figsize=(6, 4.5), dpi=300)


# Plot Train Loss
plt.plot(
    epochs,
    train_loss,
    label="Train Loss",
    # marker="o",
    linestyle="-",
    linewidth=2,
    markersize=6,
    color="tab:blue",
)

# Plot Test Loss
plt.plot(
    epochs,
    test_loss,
    label="Test Loss",
    # marker="s",
    linestyle="--",
    linewidth=2,
    markersize=6,
    color="tab:orange",
)

# Set labels and title
plt.xlabel("Epoch", fontsize=14, fontweight="bold")
plt.ylabel("Loss", fontsize=14, fontweight="bold")
plt.title("Loss Function Convergence", fontsize=16, fontweight="bold")


plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)

plt.tight_layout()

for ext in ["png", "pdf"]:
    plt.savefig(f"loss_function_convergence.{ext}", dpi=300, bbox_inches="tight")
print("Plots saved in PNG, PDF, and EPS formats.")
