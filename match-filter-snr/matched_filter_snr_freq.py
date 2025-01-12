#!/usr/bin/env python3

"""
---------------------------------------------------------------------------------------------------
ABOUT THE SCRIPT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository URL  : https://github.com/weizmannk/Virgo-DeepClean.git
Creation Date   : January 2024
Description     : This script performs matched filtering analysis on gravitational wave (GW) data
                  to evaluate signal-to-noise ratio (SNR) improvements for the pre- and post-cleaning using the
                  DeepClean algorithm. It processes cleaned GW signal data from GWF files, executes
                  matched filtering, computes Power Spectral Density (PSD), and evaluates SNR on
                  injected signals.
Usage           : python matched_filter_snr_freq.py [options]
Options         : --inj-gap <int>, --frame-org <path>, --channel-org <name>, --frame-gw-signal <path>,
                  --cleaned-data <path>, --channel-DC <name>, --channel-inj <name>, --injections-file <path>,
                  --outdir <path>, --verbose

Example         :
---------------------------------------------------------------------------------------------------
"""
# python matched_filter_snr_freq.py --inj-gap 32 --channel-inj V1:DC_INJ  --channel-DC V1:Hrec_hoft_raw_20000Hz_DC --channel-org V1:Hrec_hoft_raw_20000Hz --injections-file /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Injections/new_injection/CNR-version/142-162_Hz/injSignal/INJ-1265127585-4096.csv  --inj-gw-signal /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Injections/new_injection/CNR-version/142-162_Hz/injSignal/V1_BBH_inj_1265127585_4096.hdf5 --org-noise /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/142-162_Hz/4096/Hrec-HOFT-1265127585-50000/original-1265127585-4096.h5 --org-noise-gw-signal /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Injections/new_injection/CNR-version/142-162_Hz/noisePlusSignal/original-1265127585-4096.h5 --cleaned-noise /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/142-162_Hz/4096/Hrec-HOFT-1265127585-50000/Hrec-HOFT-1265127585-4096.gwf --cleaned-gw-signal /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Injections/new_injection/CNR-version/142-162_Hz/noisePlusSignal/Hrec-HOFT-1265127585-4096.gwf --fmin 142 --fmax 162 --outdir matchedfilter_SNR_folder/4096_32s --verbose


# Library Imports
import argparse
import os
import h5py
import pandas as pd
import numpy as np

from gwpy.timeseries import TimeSeries
from pycbc.filter import matched_filter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_time_series(file_path, channel, format="gwf"):
    """Load a TimeSeries object from a file."""
    if format == "gwf":
        return TimeSeries.read(file_path, channel)
    elif format == "h5":
        with h5py.File(file_path, "r") as file:
            data = file[channel][:]
            t0 = file[channel].attrs["t0"]
            fs = file[channel].attrs["sample_rate"]
        return TimeSeries(
            data, t0=t0, sample_rate=fs, name=channel, unit="ct", channel=channel
        )


def parse_cli_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Matched filtering on GW data for SNR analysis."
    )
    parser.add_argument(
        "--inj-gap",
        type=int,
        required=True,
        help="Time gap between injections (in seconds)",
    )

    parser.add_argument(
        "--injections-file",
        type=str,
        required=True,
        help="Path to the CSV file containing injection data",
    )
    parser.add_argument(
        "--inj-gw-signal",
        type=str,
        required=True,
        help="Path to the GW signal data frame file (HDF5 format)",
    )

    parser.add_argument(
        "--org-noise",
        type=str,
        help="Path to the original data frame file (HDF5 format)",
    )
    parser.add_argument(
        "--org-noise-gw-signal",
        type=str,
        required=True,
        help="Path to the frame file with injected GW signals",
    )

    parser.add_argument(
        "--cleaned-noise",
        type=str,
        required=True,
        help="Path to the cleaned GW data file (GWF format)",
    )
    parser.add_argument(
        "--cleaned-gw-signal",
        type=str,
        required=True,
        help="Path to the cleaned noise + GW data file (GWF format)",
    )

    parser.add_argument(
        "--channel-org",
        type=str,
        default="V1:Hrec_hoft_raw_20000Hz",
        help="Channel name for original data",
    )
    parser.add_argument(
        "--channel-DC",
        type=str,
        default="V1:Hrec_hoft_raw_20000Hz_DC",
        help="Channel name for cleaned data",
    )
    parser.add_argument(
        "--channel-inj",
        type=str,
        default="V1:DC_INJ",
        help="Channel name for injected data",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="output",
        help="Directory for saving output files and plots",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Increase output verbosity"
    )
    parser.add_argument(
        "--fmin",
        type=float,
        required=True,
        help="Minimum frequency for classification (Hz)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        required=True,
        help="Maximum frequency for classification (Hz)",
    )
    return parser.parse_args()


# Main Execution Block
if __name__ == "__main__":
    args = parse_cli_arguments()
    # Define the output directory for plots and ensure it exists
    os.makedirs(args.outdir, exist_ok=True)

    plotDir = f"{args.outdir}/Plots"
    os.makedirs(plotDir, exist_ok=True)

    injections = pd.read_csv(args.injections_file)[1:-1]
    print(len(injections))

    # Load the GW signal
    inj_gw_signal_data = TimeSeries.read(args.inj_gw_signal)

    # Load original data
    org_noise_only_data = load_time_series(
        args.org_noise, args.channel_org, format="h5"
    )
    org_noise_signal_data = load_time_series(
        args.org_noise_gw_signal, args.channel_org, format="h5"
    )

    # Load cleaned injections data
    dc_noise_only_data = load_time_series(
        args.cleaned_noise, args.channel_DC, format="gwf"
    )
    dc_noise_gw_signal_data = load_time_series(
        args.cleaned_gw_signal, args.channel_DC, format="gwf"
    )

    # Analysis logic: Matched filtering, PSD computation, SNR evaluation, and result plotting
    peak_snr_values_org = []
    time_snr_peak_org = []

    peak_snr_values_dc = []
    time_snr_peak_dc = []

    index_inj = []
    with h5py.File(args.inj_gw_signal, "r") as file:
        gw_data = file[args.channel_inj][:]
        x0 = file[args.channel_inj].attrs["x0"]
        delta_time = file[args.channel_inj].attrs["dx"]
        sample_rate = 1.0 / delta_time
        print("sample_rate : ", sample_rate)

        if args.verbose:
            print("GW signal data loaded and prepared for analysis.\n")

    for index, row in injections.iterrows():
        start_time = row["geocent_time"]
        end_time = start_time + args.inj_gap

        crop_time = 32  # so add 32 seconds then removed them in the snr
        start_index = int((start_time - x0 - crop_time) * sample_rate)
        end_index = int((end_time - x0 + crop_time) * sample_rate)

        if start_index < 0 or start_index >= end_index or start_index >= len(gw_data):
            if args.verbose:
                print(
                    f"Invalid index range for event {index}: start_index={start_index}, end_index={end_index}, len(gw_data)={len(gw_data)}"
                )
            continue

        # Convert GW signal template to TimeSeries and PyCBC TimeSeries
        gw_signal = inj_gw_signal_data[start_index:end_index]
        gw_signal_ts = gw_signal.to_pycbc()

        # PSD from orginal frame
        org_noise_only_crop = org_noise_only_data[start_index:end_index]

        psd_org = org_noise_only_crop.psd(fftlength=8, overlap=0, method="median")
        psd_ORG = psd_org.interpolate(
            org_noise_only_crop.sample_rate.value / org_noise_only_crop.size
        )
        psd_ORG = psd_ORG.to_pycbc()

        # PSD from cleaned  frame
        dc_noise_only_crop = dc_noise_only_data[start_index:end_index]

        psd_dc = dc_noise_only_crop.psd(fftlength=8, overlap=0, method="median")
        psd_DC = psd_dc.interpolate(
            dc_noise_only_crop.sample_rate.value / dc_noise_only_crop.size
        )
        psd_DC = psd_DC.to_pycbc()

        # Crop each inj_gap
        org_noise_signal_crop = org_noise_signal_data[start_index:end_index]
        dc_noise_signal_crop = dc_noise_gw_signal_data[start_index:end_index]

        # Convert gwpy Timeseries in pycbc Timesseries
        org_noise_signal_pycbc = org_noise_signal_crop.to_pycbc()
        dc_noise_signal_pycbc = dc_noise_signal_crop.to_pycbc()

        # Calculate SNR for the cleaned data using matched filtering

        # SNR from the Orginal data
        snr_org = matched_filter(
            gw_signal_ts,
            org_noise_signal_pycbc,
            psd=psd_ORG,
            low_frequency_cutoff=25,
            high_frequency_cutoff=args.fmax,
        ).crop(crop_time + 15, crop_time - 5)
        org_peak_snr_index = np.abs(snr_org).numpy().argmax()
        org_time = snr_org.sample_times[org_peak_snr_index]
        snr_org_max = np.abs(snr_org[org_peak_snr_index])
        peak_snr_values_org.append(snr_org_max)
        time_snr_peak_org.append(org_time)
        print(
            f"In the Orignal dataset:  the max of the SNR is {snr_org_max}  reach at {org_time}\n"
        )

        # SNR from the Cleaned data
        snr_dc = matched_filter(
            gw_signal_ts,
            dc_noise_signal_pycbc,
            psd=psd_DC,
            low_frequency_cutoff=25,
            high_frequency_cutoff=args.fmax,
        ).crop(crop_time + 15, crop_time - 5)
        dc_peak_snr_index = np.abs(snr_dc).numpy().argmax()
        dc_time = snr_dc.sample_times[dc_peak_snr_index]
        snr_dc_max = np.abs(snr_dc[dc_peak_snr_index])
        peak_snr_values_dc.append(snr_dc_max)
        time_snr_peak_dc.append(dc_time)
        print(
            f"In the DeepClean :  the max of the SNR is {snr_dc_max}  reach at {dc_time}\n"
        )

        # index
        index_inj.append(index)

        # Plot SNR values for original data
        plt.figure(figsize=(10, 6))
        plt.plot(snr_org.sample_times, abs(snr_org))
        plt.xlabel("Time (s)")
        plt.ylabel("SNR")
        plt.title(f"snr-{index}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plotDir,
                f"Pre_clean_snr_peaks_{int(start_time)}_{int(args.fmin)}-{int(args.fmax)}.png",
            )
        )
        plt.close()

        # Plot SNR values for cleanned data
        plt.figure(figsize=(10, 6))
        plt.plot(snr_dc.sample_times, abs(snr_dc))
        plt.xlabel("Time (s)")
        plt.ylabel("SNR")
        plt.title(f"snr-{index}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plotDir,
                f"Post_clean_snr_peaks_{int(start_time)}_{int(args.fmin)}-{int(args.fmax)}.png",
            )
        )
        plt.close()

    # save the SNR Peak for the Original data
    filename_org = f"{args.outdir}/Pre-clean_snr_peaks_{int(start_time)}_{int(args.fmin)}-{int(args.fmax)}.csv"
    data_org = {
        "Index": index_inj,
        "Peak_SNR": peak_snr_values_org,
        "Peak_SNR_Time": time_snr_peak_org,
    }
    df_org = pd.DataFrame(data_org)
    df_org.to_csv(filename_org, index=False)

    # Save the SNR Peak for the cleaned data
    filename_dc = f"{args.outdir}/Post-clean_snr_peaks_{int(start_time)}_{int(args.fmin)}-{int(args.fmax)}.csv"
    data_dc = {
        "Index": index_inj,
        "Peak_SNR": peak_snr_values_dc,
        "Peak_SNR_Time": time_snr_peak_dc,
    }
    df_dc = pd.DataFrame(data_dc)
    df_dc.to_csv(filename_dc, index=False)

    if args.verbose:
        print("SNR peaks and  plot generated and saved.")
