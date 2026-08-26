import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set figure styling
plt.rcParams["font.family"] = "serif"
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 13
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelweight"] = "bold"

# Create the output directory if it doesn't exist
outdir = "data"
os.makedirs(outdir, exist_ok=True)
directory = "data"

# GPS to UTC time difference due to leap seconds
LEAP_SECONDS = 18  # GPS time is ahead of UTC by 18 seconds (as of 2024)


def gps_to_utc(gps_time):
    """
    Convert GPS time (seconds) to UTC time, accounting for 18 seconds leap second offset.
    Args:
        gps_time (int, float, or np.int64): GPS time in seconds
    Returns:
        datetime: Corresponding UTC time as a datetime object
    """
    gps_epoch = datetime(1980, 1, 6)
    gps_time = int(gps_time)  # Use float(gps_time) if fractional seconds are needed
    utc_time = gps_epoch + timedelta(seconds=gps_time - LEAP_SECONDS)
    return utc_time


# Load the data

for freq in ["15-415_Hz"]:  #'98-110_Hz', '197-208_Hz', '142-162_Hz',
    data = pd.read_csv(os.path.join(directory, f"BNS_inspiral_range_{freq}.csv"))

    data = data.sort_values(by="Start Time", ascending=True)
    data.reset_index(drop=True, inplace=True)

    # Convert GPS times to UTC
    data["UTC Time"] = data["Start Time"].apply(gps_to_utc)

    # Data for plotting
    bns_inspiral_range_org = data["BNS Inspiral Range ORG"]
    bns_inspiral_range_dc = data["BNS Inspiral Range DC"]
    observation_index = np.arange(1, len(data) + 1)

    # Set min_time and max_time based on the first and last observations
    min_time = data["Start Time"].iloc[0]  # Start Time of the first observation
    max_time = data["Start Time"].iloc[-1]  # Start Time of the last observation
    # Revised Plotting Section

    ORG_COLOR = "green"  #'#bcbd22'  # Tab10 Olive
    DC_COLOR = "red"  #'#CC0000'   #Dark2 Purple

    PEAK_COLOR = "#003f5c"  # Dark Blue for peak annotations
    TROUGH_COLOR = "#8c564b"  # Brown for trough annotations

    fmin, fmax = int(freq.split("-")[0]), int(freq.split("-")[-1].split("_")[0])

    # Create figure and primary axis
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.set_facecolor("white")  # Use white background for clarity

    # Plotting BNS Inspiral Range before and after Deep Cleaning
    ax.plot(
        observation_index,
        bns_inspiral_range_org,
        label="V1:ORG",
        marker="o",
        linestyle="-",
        color=ORG_COLOR,
    )
    ax.plot(
        observation_index,
        bns_inspiral_range_dc,
        label="V1:DC",
        marker="s",
        linestyle="--",
        color=DC_COLOR,
    )

    # Peak Detection
    peak_prominence = 2  # Adjust based on your data's characteristics
    peaks, properties = find_peaks(bns_inspiral_range_dc, prominence=peak_prominence)

    # Annotate each peak with its UTC time in bold font
    for peak in peaks:
        utc_time = data["UTC Time"].iloc[peak]
        formatted_time = utc_time.strftime("%d/%m/%Y, %H:%M")
        ax.annotate(
            formatted_time,
            (observation_index[peak], bns_inspiral_range_dc.iloc[peak]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=12,
            fontweight="bold",
            color="#003f5c",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
        )

    # Trough Detection (Low Peaks)
    # Invert the data to find troughs
    inverted_bns_inspiral_range_dc = -bns_inspiral_range_dc

    # Define prominence for trough detection
    trough_prominence = 1  # Adjust based on your data's characteristics
    troughs, trough_properties = find_peaks(
        inverted_bns_inspiral_range_dc, prominence=trough_prominence
    )

    if fmin == 15:
        offset_value = -70
    else:
        offset_value = -40

    # Annotate each trough with its UTC time in bold font
    for trough in troughs:
        utc_time = data["UTC Time"].iloc[trough]
        formatted_time = utc_time.strftime("%d/%m/%Y, %H:%M")
        ax.annotate(
            formatted_time,
            (observation_index[trough], bns_inspiral_range_dc.iloc[trough]),
            textcoords="offset points",
            xytext=(0, offset_value),  # Offset label below the trough
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=TROUGH_COLOR,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
        )

    if fmin == 15:
        # Add frequency band text with bold font in a textbox
        plt.text(
            0.02,
            0.3,
            f"Frequency band: {fmin} to {fmax} Hz",
            transform=ax.transAxes,
            fontsize=12,
            color="#003f5c",
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

    else:  # Add frequency band text with bold font in a textbox
        plt.text(
            0.02,
            0.38,
            f"Frequency band: {fmin} to {fmax} Hz",
            transform=ax.transAxes,
            fontsize=12,
            color="#003f5c",
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

    # Labels and title
    ax.set_xlabel(r"Observation Index", fontsize=15, fontweight="bold")
    ax.set_ylabel(r"BNS Inspiral Range (Mpc)", fontsize=15, fontweight="bold")
    ax.legend(loc="lower left", framealpha=1, fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add a secondary X-axis for GPS Time
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())  # Align the secondary axis with the primary axis

    # Define the number of GPS time labels
    num_labels = min(4, len(observation_index))  # Adjust based on data length
    tick_indices = np.linspace(1, len(observation_index), num=num_labels, dtype=int)
    tick_labels = [data["Start Time"].iloc[i - 1] for i in tick_indices]

    # Configure the secondary X-axis
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(
        tick_labels, rotation=0, fontsize=12, fontweight="bold", color="black"
    )
    ax2.set_xlabel("GPS Time", fontsize=14, fontweight="bold", color="black")

    # Adjust y-limits for better visualization
    if fmin == 15:
        ax.set_ylim(
            np.min(bns_inspiral_range_dc) - 2, np.max(bns_inspiral_range_dc) + 0.6
        )
    else:
        ax.set_ylim(
            np.min(bns_inspiral_range_dc) - 0.5, np.max(bns_inspiral_range_dc) + 0.5
        )

    # Save and show plot
    output_path = os.path.join(
        outdir, f"BNS_Inspiral_Range_Evolution_{fmin}_{fmax}.pdf"
    )
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
