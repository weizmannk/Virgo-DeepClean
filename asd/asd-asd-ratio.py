#!/usr/bin/env python

"""
---------------------------------------------------------------------------------------------------
ABOUT THE SCRIPT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository URL  : https://github.com/weizmannk/Virgo-DeepClean.git
Creation Date   : January 2024
Description     : This Python script is designed for analyzing gravitational wave data by generating
                  Amplitude Spectral Density (ASD) and ASD ratio plots. It is optimized for datasets
                  both pre and post-processing with the DeepClean algorithm. The script allows users to
                  specify data directories, frequency ranges for analysis, and additional plotting
                  parameters via command-line arguments. The initial concept was developed by Saleem and
                  later adapted and extended by Weizmann for broader applications.
Usage           : python asd-asd-ratio.py --path-dir /path/to/data --outdir /path/to/output --fmin <freq min> --fmax <freq max>

Example         : python asd-asd-ratio.py --path-dir /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/retrain/4096/layer14  --outdir ./Plots  --fmin 15 --fmax 415  --custom-ticks  15,50,100,150,200,300,415
---------------------------------------------------------------------------------------------------
"""


import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from gwpy.timeseries import TimeSeries

from matplotlib.gridspec import GridSpec

# Configure matplotlib to use 'Agg' backend
plt.switch_backend("agg")

# Initialize logging
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_arguments():
    """
    Parse command-line arguments.
    Returns:
        args: ArgumentParser object with command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate ASD and ASD ratio plots for GW data training."
    )
    parser.add_argument(
        "--path-dir", type=str, required=True, help="Path to data directory"
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Output directory for plots"
    )
    parser.add_argument(
        "--channel-org",
        type=str,
        default="V1:Hrec_hoft_raw_20000Hz",
        help="Original channel",
    )
    parser.add_argument(
        "--channel-dc",
        type=str,
        default="V1:Hrec_hoft_raw_20000Hz_DC",
        help="DeepClean channel",
    )
    parser.add_argument("--fmin", type=float, default=142, help="Minimum frequency")
    parser.add_argument("--fmax", type=float, default=162, help="Maximum frequency")
    parser.add_argument(
        "--custom-ticks",
        type=str,
        help='Comma-separated list of custom x-axis tick values (e.g., "10,20,400")',
    )

    return parser.parse_args()


# def plot_asd_and_ratio(data_org, data_dc, figname, start, end, fmin, fmax):
#     """
#     Plot ASD and ASD ratio in a single figure with two subplots.
#     Parameters:
#         data_org: TimeSeries - Original data series
#         data_dc: TimeSeries - DeepClean processed data series
#         figname: str - Filename for the saved figure
#         start: int - Start time of the data
#         end: int - End time of the data
#         fmin: float - Minimum frequency
#         fmax: float - Maximum frequency
#     """
#     asd_org = data_org.asd(fftlength=8, overlap=0, method='median').crop(fmin, fmax)
#     asd_dc = data_dc.asd(fftlength=8, overlap=0, method='median').crop(fmin, fmax)

#     ratio = asd_dc / asd_org

#     fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

#     # Plot ASD
#     axs[0].loglog(asd_org, label=r'V1:ORG', linewidth=1.5, color='green')
#     axs[0].loglog(asd_dc, label=r'V1:DC', linewidth=1.5, color='red')
#     axs[0].set_ylabel('ASD [Hz$^{-1/2}$]')
#     axs[0].set_xlim(fmin, fmax)
#     axs[0].set_ylim(1e-23, 1e-20/2)
#     axs[0].legend(loc='best')
#     axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

#     # Plot ASD Ratio
#     axs[1].loglog(ratio, color='blue', linewidth=1.5)
#     axs[1].set_xlabel('Frequency (Hz)')
#     axs[1].set_ylabel('ASD Ratio')
#     axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
#     axs[1].set_xlim(fmin, fmax)
#     axs[1].set_ylim(0.01, 1.5)

#     fig.suptitle(f'ASD and ASD Ratio from {start} to {end}')
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig(figname, dpi=300)
#     logging.info(f'Plot saved to {figname}')
#     plt.close(fig)  # Close the figure to free up memory


def plot_asd_and_ratio(
    data_org, data_dc, figname, start, end, fmin, fmax, custom_ticks
):
    """
    Plot ASD and ASD ratio in a single figure with two subplots of different sizes.
    Parameters:
        data_org: TimeSeries - Original data series
        data_dc: TimeSeries - DeepClean processed data series
        figname: str - Filename for the saved figure
        start: int - Start time of the data
        end: int - End time of the data
        fmin: float - Minimum frequency
        fmax: float - Maximum frequency
        custom_ticks: str - Comma-separated string of custom x-axis tick values
    """
    # Convert the custom_ticks from a comma-separated string to a list of integers
    if custom_ticks:
        custom_tick_values = [int(tick) for tick in custom_ticks.split(",")]
    else:
        custom_tick_values = []

    asd_org = data_org.asd(fftlength=8, overlap=0, method="median").crop(fmin, fmax)
    asd_dc = data_dc.asd(fftlength=8, overlap=0, method="median").crop(fmin, fmax)
    ratio = asd_dc / asd_org

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    ax1 = fig.add_subplot(gs[0])
    ax1.loglog(asd_org, label=r"V1:ORG", linewidth=1.5, color="green")
    ax1.loglog(asd_dc, label=r"V1:DC", linewidth=1.5, color="red")
    ax1.set_ylabel("ASD [Hz$^{-1/2}$]")
    ax1.set_xlim(fmin, fmax)
    ax1.set_ylim(1e-23, 1e-20 / 2)
    if custom_tick_values:
        ax1.set_xticks(custom_tick_values)
        ax1.set_xticklabels([str(tick) for tick in custom_tick_values])
    ax1.legend(loc="best")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    ax2 = fig.add_subplot(gs[1])
    ax2.loglog(ratio, color="blue", linewidth=1.5)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("ASD Ratio")
    ax2.set_xlim(fmin, fmax)
    ax2.set_ylim(0.01, 1.5)
    if custom_tick_values:
        ax2.set_xticks(custom_tick_values)
        ax2.set_xticklabels([str(tick) for tick in custom_tick_values])
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.suptitle(f"ASD and ASD Ratio from {start} to {end}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figname, dpi=300)
    logging.info(f"Plot saved to {figname}")
    plt.close(fig)


def generate_asd_plots(args):
    """
    Generate ASD and ASD ratio plots for all files in the specified directory.
    Parameters:
        args: ArgumentParser object with command-line arguments
    """

    os.makedirs(args.outdir, exist_ok=True)
    locs = glob.glob(os.path.join(args.path_dir, "Hrec*"), recursive=True)

    for loc in locs:
        frame_dc_files = glob.glob(os.path.join(loc, "Hrec*.gwf"))
        frame_org_files = glob.glob(os.path.join(loc, "orig*h5"))
        # frame_org_files = glob.glob(os.path.join(dir_org, "orig*h5"))

        for frame_dc, frame_org in zip(sorted(frame_dc_files), sorted(frame_org_files)):
            start, duration = map(
                int, (frame_org.split("original-")[1].split(".h5")[0].split("-")[:2])
            )
            data_org = timeSeries_from_h5(frame_org, args.channel_org)
            data_dc = TimeSeries.read(frame_dc, args.channel_dc)
            figname = os.path.join(
                args.outdir,
                f"asd_comparison_{start}-{duration}_{args.fmin}-{args.fmax}_Hz.pdf",
            )
            plot_asd_and_ratio(
                data_org,
                data_dc,
                figname,
                start,
                start + duration,
                args.fmin,
                args.fmax,
                args.custom_ticks,
            )


def timeSeries_from_h5(filename, channel="V1:Hrec_hoft_raw_20000Hz"):
    """
    Read .h5 output files produced by dc-prod-clean.
    Parameters:
        filename: str - The path to the .h5 file
        channel: str - Channel name to extract data from
    Returns:
        TimeSeries object with the data, sample rate, and other attributes
    """
    with h5py.File(filename, "r") as f:
        data = f[channel][:]
        t0 = f[channel].attrs["t0"]
        fs = f[channel].attrs["sample_rate"]
    data_ts = TimeSeries(
        data, t0=t0, sample_rate=fs, name=channel, unit="ct", channel=channel
    )
    return data_ts


if __name__ == "__main__":
    args = parse_arguments()
    generate_asd_plots(args)
