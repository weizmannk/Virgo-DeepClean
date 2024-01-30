#!/usr/bin/env python3

"""
---------------------------------------------------------------------------------------------------
ABOUT THE SCRIPT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository URL  : https://github.com/weizmannk/ObservingScenariosInsights.git
Creation Date   : January 2024
Description     : This Python script performs matched filtering analysis on gravitational wave (GW) data
                  to assess the SNR prior to the application of the DeepClean algorithm. It is designed
                  to load GW signal data, calculate Power Spectral Density (PSD), and process signal
                  injections with flexibility through command-line arguments for file paths, channels,
                  and injection parameters. This script facilitates the pre-cleaning evaluation of GW
                  data across various datasets and research scenarios, contributing to the advancement
                  of gravitational wave studies.
Usage           : python match_filter_snr_pre_clean.py --inj-gap <int> --frame-org <path> --channel-org <name>
                  --frame-inj <path> --frame-gw-signal <path> --channel-inj <name> --injections-file <path>
                  --verbose (optional flag for verbose output)
                  
example         :  python match_filter_snr_pre_clean.py --inj-gap 32  --channel-org V1:Hrec_hoft_raw_20000Hz --frame-inj injected_original-1265127585-4096.h5 --frame-gw-signal V1_BBH_inj_1265127585.0_1265131681.0.hdf5  --channel-inj V1:DC_INJ --injections-file INJ-1265127585-4096.csv --verbose 
---------------------------------------------------------------------------------------------------
"""

# Library Imports
import argparse
import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from pycbc.filter import matched_filter, resample_to_delta_t, highpass
from pycbc.psd import interpolate, inverse_spectrum_truncation

# Function Definitions

def load_time_series_from_h5(file_path, channel):
    """Generates a TimeSeries object from an H5 file containing PSD data."""
    with h5py.File(file_path, 'r') as file:
        if 't0' not in file[channel].attrs:
            raise KeyError(f"'t0' attribute missing in channel {channel}.")
        data = file[channel][:]
        t0 = file[channel].attrs['t0']
        fs = file[channel].attrs['sample_rate']
    return TimeSeries(data, t0=t0, sample_rate=fs, name=channel, unit="ct", channel=channel)

def parse_cli_arguments():
    """Parses command-line arguments for configuring the analysis."""
    parser = argparse.ArgumentParser(description="Matched Filtering Analysis Pre-DeepClean.")
    parser.add_argument('--inj-gap', type=int, required=True, help='Time gap between signal injections (in seconds)')
    parser.add_argument('--frame-org', type=str, help='Path to the original GW data frame file (HDF5 format)')
    parser.add_argument('--frame-inj', type=str, required=True, help='Path to the frame file with injected GW signals')
    parser.add_argument('--frame-gw-signal', type=str, required=True, help='Path to the GW signal data frame file (HDF5 format)')
    parser.add_argument('--injections-file', type=str, required=True, help='CSV file containing injection data')
    parser.add_argument('--channel-org', type=str, default='V1:Hrec_hoft_raw_20000Hz', help='Channel name for original GW data')
    parser.add_argument('--channel-inj', type=str, default='V1:DC_INJ', help='Channel name for injected GW signals')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--outdir', type=str, default='output', help='Output directory for plots and analysis results')
    return parser.parse_args()

# Main Execution Block

if __name__ == "__main__":
    args = parse_cli_arguments()
    os.makedirs(args.outdir, exist_ok=True)  # Ensure the output directory exists

    # Load original and injected data for analysis
    if args.frame_org:
        original_data = load_time_series_from_h5(args.frame_org, args.channel_org)
    injected_data = load_time_series_from_h5(args.frame_inj, args.channel_org)
    
    # Load injections data from CSV and exclude first and last events for better PSD and SNR results
    injections_data = pd.read_csv(args.injections_file).iloc[1:-1]

    crop_time = args.inj_gap / 2

    # Load GW signal data for processing
    with h5py.File(args.frame_gw_signal, 'r') as file:
        gw_signal_data = file[args.channel_inj][:]
        start_time = file[args.channel_inj].attrs['x0']
        delta_time = file[args.channel_inj].attrs['dx']
        sample_rate = 1.0 / delta_time

        # Convert to TimeSeries and PyCBC types for analysis
        gw_signal_ts = TimeSeries(gw_signal_data, t0=start_time, sample_rate=sample_rate, name=args.channel_inj, unit="ct", channel=args.channel_inj).to_pycbc()
        filtered_gw_signal = highpass(gw_signal_ts, 15.0)
        filtered_injected_data = highpass(injected_data.to_pycbc(), 15.0)

        # Resample and crop signals for consistent analysis duration
        resampled_gw_signal = resample_to_delta_t(filtered_gw_signal, 1/sample_rate).crop(crop_time, crop_time)
        resampled_injected_data = resample_to_delta_t(filtered_injected_data, 1/sample_rate).crop(crop_time, crop_time)

        # Compute PSD for the injected data segment
        psd = resampled_injected_data.psd(8)
        psd = interpolate(psd, resampled_injected_data.delta_f)
        psd = inverse_spectrum_truncation(psd, int( 8 * resampled_injected_data.sample_rate), low_frequency_cutoff=20.0)

        # Calculate SNR using matched filtering
        snr = matched_filter(resampled_gw_signal, resampled_injected_data, psd=psd, low_frequency_cutoff=25)
        snr = snr.crop(crop_time, crop_time)  # Crop to avoid edge effects
        
        peak_snr_values = []
        time_snr_peak = []
        num_inj = []
        n_inj = len(injections_data)
        for index, row in injections_data.iterrows():
            if args.verbose:
                print(f"Processing injection {index}/{n_inj}")
            t0 = row['geocent_time']
            t1 = row['geocent_time'] + args.inj_gap
            start_index = int((t0 - start_time) * sample_rate)
            end_index = int((t1 - start_time) * sample_rate)

            # Error checking for indices
            if start_index < 0 or end_index > len(snr):
                if args.verbose:
                    print(f"Error: Indices out of bounds for SNR array. Start: {start_index}, End: {end_index}")
                continue  # Skip this iteration and continue with the next

            snr_select = snr[start_index:end_index]
            if len(snr_select) == 0:
                if args.verbose:
                    print(f"No data found in SNR array for indices {start_index} to {end_index}.")
                continue  # Skip this iteration and continue with the next

            peak_snr = abs(snr_select).numpy().argmax()
            time = snr_select.sample_times[peak_snr]
            peak_snr_values.append(abs(snr_select[peak_snr]))
            time_snr_peak.append(time)
            num_inj.append(index)

            if args.verbose:
                print(f"Peak SNR: {abs(snr_select[peak_snr])} at time {time}")

        # Plot and save SNR results
        plt.figure(figsize=(10, 6))
        plt.scatter(time_snr_peak, peak_snr_values, label='Pre-Clean SNR')
        plt.xlabel('Time (s)')
        plt.ylabel('SNR')
        plt.title('SNR Evaluation Pre-Cleaning')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.outdir, 'pre_clean_snr_plot.png'))
        plt.show()

        # Save SNR data for further analysis
        snr_data = { "Injection-num" :num_inj, 'Time': time_snr_peak, 'SNR': peak_snr_values}
        snr_df = pd.DataFrame(snr_data)
        snr_df.to_csv(os.path.join(args.outdir, 'snr_values_pre_clean.csv'), index=False)

        if args.verbose:
            print(f"Pre-cleaning SNR analysis completed. Results are stored in '{args.outdir}'.")
