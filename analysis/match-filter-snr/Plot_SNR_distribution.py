import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import norm
from matplotlib.lines import Line2D

# Configure plot aesthetics
plt.rcParams["font.family"] = "serif"
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["ytick.labelsize"] = 22
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["legend.fontsize"] = 22
plt.rcParams["axes.titlesize"] = 2
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True  # Ensure LaTeX is enabled

# Create the output directory if it doesn't exist
outdir = "Plots"
os.makedirs(outdir, exist_ok=True)
directory = "SNR-Peak"

# Define the threshold for SNR fractional_difference
threshold = 0.01

for freq in ["98-110_Hz", "142-162_Hz", "197-208_Hz", "15-415_Hz"]:
    print("*********************************************")
    print(f"\nFrequency Band: {freq}\n")

    # Load the data
    snr_pre_clean = pd.read_csv(os.path.join(directory, f"Pre_SNR_all_{freq}_32s.csv"))
    snr_post_clean = pd.read_csv(
        os.path.join(directory, f"Post_SNR_all_{freq}_32s.csv")
    )

    # Sort by peak SNR time
    snr_pre_clean = snr_pre_clean.sort_values(by="Peak_SNR_Time", ascending=True)
    snr_post_clean = snr_post_clean.sort_values(by="Peak_SNR_Time", ascending=True)

    # Calculate fractional differences between pre and post DeepClean SNRs
    fractional_difference = (
        (snr_post_clean["Peak_SNR"] - snr_pre_clean["Peak_SNR"])
        / snr_pre_clean["Peak_SNR"]
    ) * 100

    # Apply filtering based on the threshold
    unchanged_indices = fractional_difference[
        np.abs(fractional_difference) <= threshold
    ].index
    increased_indices = fractional_difference[fractional_difference > threshold].index
    decreased_indices = fractional_difference[fractional_difference < -threshold].index

    # Calculate the percentages
    percentage_unchanged = len(unchanged_indices) / len(fractional_difference) * 100
    percentage_increased = len(increased_indices) / len(fractional_difference) * 100
    percentage_decreased = len(decreased_indices) / len(fractional_difference) * 100

    # Print results
    print(f"Threshold for unchanged SNR: ±{threshold}")
    print(
        f"Unchanged SNRs (|fractional_difference| ≤ {threshold}%): {percentage_unchanged:.2f}%"
    )
    print(
        f"Increased SNRs (fractional_difference > {threshold}%): {percentage_increased:.2f}%"
    )
    print(
        f"Decreased SNRs (fractional_difference < -{threshold}%): {percentage_decreased:.2f}%"
    )

    # Optional: Save filtered data
    unchanged_snr = snr_pre_clean.loc[unchanged_indices]
    increased_snr = snr_pre_clean.loc[increased_indices]
    decreased_snr = snr_pre_clean.loc[decreased_indices]

    # Calculate statistics for histogram
    mean_fractional_difference = fractional_difference.mean()
    std_fractional_difference = fractional_difference.std()
    median_fractional_difference = fractional_difference.median()

    # Generate points on the x-axis for the normal distribution curve
    x_points = np.linspace(
        mean_fractional_difference - 5 * std_fractional_difference,
        mean_fractional_difference + 5 * std_fractional_difference,
        10000,
    )
    y_points = norm.pdf(x_points, mean_fractional_difference, std_fractional_difference)

    # Calculate the number of data points within +/- 1 standard deviations
    within_1_std = fractional_difference[
        (
            fractional_difference
            >= mean_fractional_difference - 1 * std_fractional_difference
        )
        & (
            fractional_difference
            <= mean_fractional_difference + 1 * std_fractional_difference
        )
    ]

    # Define colors for different categories
    color_increased = "#9400D3"  # Purple for increased
    color_decreased = "#ff7f0e"  # Orange for decreased
    color_unchanged = "olive"  # Olive for unchanged

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

    delta_snr = r"$\frac{\Delta \mathrm{SNR}}{\mathrm{SNR}}$"
    percent = r"$(\%)$"
    # Plot histogram
    sns.histplot(
        fractional_difference,
        bins=30,
        kde=False,
        stat="density",
        color="gray",
        alpha=0.6,
        label=f"{delta_snr} distribution",
        ax=ax,
    )

    # Add jitter for scatter points
    jitter = np.abs(np.random.normal(scale=0.1, size=len(fractional_difference)))

    # Scatter plots for different categories
    ax.scatter(
        fractional_difference.loc[decreased_indices],
        jitter[decreased_indices],
        color=color_decreased,
        alpha=0.6,
        label="Changed (Decreased)",
        s=12,
    )
    ax.scatter(
        fractional_difference.loc[unchanged_indices],
        jitter[unchanged_indices],
        color=color_unchanged,
        alpha=0.6,
        label="Unchanged",
        s=12,
    )
    ax.scatter(
        fractional_difference.loc[increased_indices],
        jitter[increased_indices],
        color=color_increased,
        alpha=0.6,
        label="Changed (Increased)",
        s=12,
    )

    # Plot the normal distribution curve
    ax.plot(
        x_points, y_points, color="darkblue", label=r"$\mathcal{N}(\mu, \sigma^{2})$"
    )

    # Plot mean and standard deviation lines
    ax.axvline(
        mean_fractional_difference,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=rf"$\mu$: {mean_fractional_difference:.4f}\%",
    )
    ax.axvline(
        mean_fractional_difference + std_fractional_difference,
        color="red",
        linestyle="dotted",
        linewidth=2,
        label=rf"$\sigma$: {std_fractional_difference:.4f}\%",
    )
    ax.axvline(
        mean_fractional_difference - std_fractional_difference,
        color="red",
        linestyle="dotted",
        linewidth=2,
    )

    # Set labels with LaTeX formatting
    ax.set_xlabel(f"Peak of {delta_snr} distribution {percent}", fontsize=24)
    ax.set_ylabel("Probability Density", fontsize=24)

    # Create custom legends
    individual_diff_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Decreased",
            markerfacecolor=color_decreased,
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Unchanged",
            markerfacecolor=color_unchanged,
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Increased",
            markerfacecolor=color_increased,
            markersize=8,
        ),
    ]

    other_legend = [
        Line2D([0], [0], color="darkblue", label=r"$\mathcal{N}(\mu, \sigma^{2})$"),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=rf"$\mu$: {mean_fractional_difference:.1f}\%",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="dotted",
            linewidth=2,
            label=rf"$\sigma$: {std_fractional_difference:.1f}\%",
        ),
        Line2D([0], [0], color="gray", lw=10, label=f"{delta_snr} distribution"),
    ]

    # Add legends to the plot based on frequency band
    fmin, fmax = int(freq.split("-")[0]), int(freq.split("-")[-1].split("_")[0])

    # Add the individual differences legend
    legend1 = ax.legend(
        handles=individual_diff_legend,
        title=f"{delta_snr}",
        loc="upper right",
        title_fontsize=22,
        prop={"size": 20},
    )
    ax.add_artist(legend1)  # Add the first legend

    # Add the other legend
    ax.legend(handles=other_legend, loc="upper left", prop={"size": 20})

    # Add frequency band annotation
    ax.text(
        0.02,
        0.2,
        f"Frequency band: {fmin} to {fmax} Hz",
        transform=ax.transAxes,
        fontsize=20,
        color="#003f5c",
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    # Save the plot
    plt.savefig(
        os.path.join(outdir, f"SNR_Hist_different_{freq}.pdf"),
        bbox_inches="tight",
        dpi=800,
    )
    plt.close()

    # Print summary
    print("==============================\n")
    print(f"Increased: {percentage_increased:.2f}%")
    print(f"Decreased: {percentage_decreased:.2f}%")
    print(f"Unchanged: {percentage_unchanged:.2f}%")
    print("+++++++++++++++++++++++++++++++++++++++++\n")

    print("==============================\n")
    print(f"Median Fractional Difference: {median_fractional_difference:.2f}%")
    print(f"Standard Deviation: {std_fractional_difference:.2f}%")
    # print(f"Within {sigma_value} standard deviations: {percentage_within_5_std:.2f}%")

    improvement = fractional_difference.mean()

    print("==============================\n")
    print(f"Improvement: {improvement:.2f}%")
    print("...............................\n")

    print("+++++++++++++++++++++++++++++++++++++++++\n")

    print("min of sigma :", min(within_1_std))
    print("max of sigma :", max(within_1_std))
    print(mean_fractional_difference + std_fractional_difference)
