import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import norm

# Configure plot aesthetics
plt.rcParams["font.family"] = "serif"
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelweight"] = "bold"

# Create the output directory if it doesn't exist
outdir = "Plots"
os.makedirs(outdir, exist_ok=True)
directory = "SNR-Peak"

# Define the threshold for SNR difference
threshold = 0.05

for freq in ["98-110_Hz", "142-162_Hz", "197-208_Hz", "15-415_Hz"]:
    print("*********************************************")
    print(f"\n {freq}\n")

    # Load the data
    snr_pre_clean = pd.read_csv(os.path.join(directory, f"Pre_SNR_all_{freq}_32s.csv"))
    snr_post_clean = pd.read_csv(
        os.path.join(directory, f"Post_SNR_all_{freq}_32s.csv")
    )

    # Sort by peak SNR time
    snr_pre_clean = snr_pre_clean.sort_values(by="Peak_SNR_Time", ascending=True)
    snr_post_clean = snr_post_clean.sort_values(by="Peak_SNR_Time", ascending=True)

    # Calculate the differences between pre and post DeepClean SNRs
    difference = snr_post_clean["Peak_SNR"] - snr_pre_clean["Peak_SNR"]

    # Apply filtering based on the threshold
    unchanged_indices = difference[np.abs(difference) <= threshold].index
    increased_indices = difference[difference > threshold].index
    decreased_indices = difference[difference < -threshold].index

    # Calculate the percentages
    percentage_unchanged = len(unchanged_indices) / len(difference) * 100
    percentage_increased = len(increased_indices) / len(difference) * 100
    percentage_decreased = len(decreased_indices) / len(difference) * 100

    # Print results
    print(f"Threshold for unchanged SNR: ±{threshold}")
    print(f"Unchanged SNRs (|difference| <= {threshold}): {percentage_unchanged:.2f}%")
    print(f"Increased SNRs (difference > {threshold}): {percentage_increased:.2f}%")
    print(f"Decreased SNRs (difference < -{threshold}): {percentage_decreased:.2f}%")

    # Optional: Save filtered data
    unchanged_snr = snr_pre_clean.loc[unchanged_indices]
    increased_snr = snr_pre_clean.loc[increased_indices]
    decreased_snr = snr_pre_clean.loc[decreased_indices]

    # Calculate statistics for histogram
    mean_difference = difference.mean()
    std_difference = difference.std()
    median_difference = difference.median()

    # Generate points on the x-axis for the normal distribution curve
    x_points = np.linspace(
        mean_difference - 5 * std_difference,
        mean_difference + 5 * std_difference,
        10000,
    )
    y_points = norm.pdf(x_points, mean_difference, std_difference)

    # Define colors for different categories
    color_increased = "#9400D3"  # Green for increased
    color_decreased = "#ff7f0e"  # Orange for decreased
    color_unchanged = "olive"  # Olive for unchanged

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    sns.histplot(
        difference,
        bins=30,
        kde=False,
        stat="density",
        color="gray",
        alpha=0.6,
        label="SNR difference distribution",
        ax=ax,
    )
    jitter = np.random.normal(scale=0.1, size=len(difference))
    # ax.scatter(difference, jitter, color='#9400D3', alpha=0.7, s=20, label='Individual differences')

    # Scatter plots for different categories
    scatter_decreased = ax.scatter(
        difference.loc[decreased_indices],
        jitter[decreased_indices],
        color=color_decreased,
        alpha=0.6,
        label="Changed (Decreased)",
        s=12,
    )
    scatter_unchanged = ax.scatter(
        difference.loc[unchanged_indices],
        jitter[unchanged_indices],
        color=color_unchanged,
        alpha=0.6,
        label="Unchanged",
        s=12,
    )
    scatter_changed = ax.scatter(
        difference.loc[increased_indices],
        jitter[increased_indices],
        color=color_increased,
        alpha=0.6,
        label="Changed (Increased)",
        s=12,
    )

    ax.plot(
        x_points, y_points, color="darkblue", label="$\mathcal{N}(\mu,\,\sigma^{2})$"
    )
    ax.axvline(
        mean_difference,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"$\\mu$: {mean_difference:.4f}",
    )
    ax.axvline(
        mean_difference + std_difference,
        color="red",
        linestyle="dotted",
        linewidth=2,
        label=f"$\\sigma$: {std_difference:.4f}",
    )
    ax.axvline(
        mean_difference - std_difference, color="red", linestyle="dotted", linewidth=2
    )

    # Annotate the plot
    ax.set_xlabel(r"Peak SNR difference", fontsize=18)
    ax.set_ylabel(r"Probability density", fontsize=18)

    # ax.text(ax.get_xlim()[0] * 0.9, ax.get_ylim()[1] * 0.3, f'{len(difference[(np.abs(difference) <= threshold)]):.2f}% within threshold', fontsize=12, color='navy')

    # Create custom legend for individual differences
    from matplotlib.lines import Line2D

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

    # Create custom legend for other plot elements
    other_legend = [
        Line2D([0], [0], color="darkblue", label="$\mathcal{N}(\mu,\,\sigma^{2})$"),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"$\\mu$: {mean_difference:.2f}",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="dotted",
            linewidth=2,
            label=f"$\\sigma$: {std_difference:.2f}",
        ),
        Line2D([0], [0], color="gray", lw=8, label="SNR difference distribution"),
    ]

    # Calculate the number of data points within +/- 5 standard deviations
    sigma_value = 5
    within_5_std = difference[
        (difference >= mean_difference - sigma_value * std_difference)
        & (difference <= mean_difference + sigma_value * std_difference)
    ]

    percentage_within_5_std = len(within_5_std) / len(difference) * 100

    cred_interval = 0.9
    z_value = norm.ppf(cred_interval, mean_difference, std_difference)

    width = difference[
        (difference >= mean_difference - z_value * std_difference)
        & (difference <= mean_difference + z_value * std_difference)
    ]

    cred_per = len(width) / len(difference) * 100
    print("==============================\n")
    print("value :", z_value)
    print("...............................\n")
    print("min of sigma :", min(width))
    print("max of sigma :", max(width))

    print(
        f"\n{cred_interval * 100} % of SNR difference are lie from {min(width)} to {max(width)}"
    )
    print("==============================\n")

    fmin, fmax = int(freq.split("-")[0]), int(freq.split("-")[-1].split("_")[0])

    if fmin == 15:
        # Add the legends to the plot
        legend1 = ax.legend(
            handles=individual_diff_legend,
            title=r"SNR differences",
            loc="upper left",
            title_fontsize=18,
            prop={"size": 17},
        )
        ax.add_artist(legend1)  # Add the first legend

        ax.legend(handles=other_legend, loc="upper right", prop={"size": 17})

        ax.text(
            0.56,
            0.08,
            f"Frequency band: {fmin} to {fmax} Hz",
            transform=ax.transAxes,
            fontsize=16,
            color="#003f5c",
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        # position fo text
        x_text = ax.get_xlim()[1] * 0.7
        y_text = ax.get_ylim()[1] * 0.2

    else:
        # Add the legends to the plot
        legend1 = ax.legend(
            handles=individual_diff_legend,
            title=r"SNR differences",
            loc="upper right",
            title_fontsize=18,
            prop={"size": 17},
        )
        ax.add_artist(legend1)  # Add the first legend,

        ax.legend(handles=other_legend, loc="upper left", prop={"size": 17})

        ax.text(
            0.02,
            0.06,
            f"Frequency band: {fmin} to {fmax} Hz",
            transform=ax.transAxes,
            fontsize=16,
            color="#003f5c",
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

    #          # position fo text
    #         x_text = ax.get_xlim()[0] * 0.9
    #         y_text = ax.get_ylim()[1] * 0.3

    # # Add text
    # ax.text(x_text, y_text, f'{percentage_within_5_std:.2f} % within 5$\sigma$', fontsize=12, color='navy')

    # Save the plot
    plt.savefig(
        os.path.join(outdir, f"SNR_Hist_different_{freq}.pdf"),
        bbox_inches="tight",
        dpi=800,
    )
    plt.close()

    print("==============================\n")
    print("Increasement: ", percentage_increased, "%")
    print("Decrease: ", percentage_decreased, "%")
    print("Equality: ", percentage_unchanged, "%")
    print("+++++++++++++++++++++++++++++++++++++++++\n")

    print("==============================\n")
    print("median: ", mean_difference)
    print("standar deviation: ", std_difference)
    # print(f"Within {sigma_value} standar deviation: ",  percentage_within_5_std, "%")

    improve = (difference / snr_pre_clean["Peak_SNR"]).mean() * 100

    print("==============================\n")
    print("Improvement :", improve, "%")
    print("...............................\n")
