#!/usr/bin/env python3

"""
---------------------------------------------------------------------------------------------------
ABOUT THE SCRIPT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository URL  : https://github.com/weizmannk/ObservingScenariosInsights.git
Creation Date   : January 2024
Description     : This Python script implements matched filtering analysis on gravitational wave (GW)
                  data to evaluate signal-to-noise ratio (SNR) improvements post-cleaning using the
                  DeepClean algorithm. It processes cleaned GW signal data from GWF files, executes
                  matched filtering, computes Power Spectral Density (PSD), and evaluates SNR on
                  injected signals. The script is designed for use in scientific workflows to verify
                  GW signal integrity after cleaning.
Usage           : python match_filter_snr_post_clean.py [options]
Options         : --inj-gap <int>, --frame-org <path>, --channel-org <name>, --frame-gw-signal <path>,
                  --cleaned-data <path>, --channel-DC <name>, --channel-inj <name>, --injections-file <path>,
                  --outdir <path>, --verbose
Example         : python match_filter_snr_post_clean.py --inj-gap 32 --channel-org V1:Hrec_hoft_raw_20000Hz --frame-gw-signal V1_BBH_inj_1265127585.0_1265131681.0.hdf5 --cleaned-data  injected_clean_Hrec-HOFT-1265127585-4096.gwf --channel-DC V1:Hrec_hoft_raw_20000Hz_DC --channel-inj V1:DC_INJ --injections-file INJ-1265127585-4096.csv --verbose  
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
def load_time_series_from_gwf(file_path, channel):
    """Generates a TimeSeries object from a GWF file."""
    return TimeSeries.read(file_path, channel)

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
    """Parses command-line arguments provided to the script."""
    parser = argparse.ArgumentParser(description="Performs matched filtering on GW data for SNR analysis.")
    parser.add_argument('--inj-gap', type=int, required=True, help='Time gap between injections (in seconds)')
    parser.add_argument('--frame-org', type=str, help='Path to the original data frame file (HDF5 format)')
    parser.add_argument('--frame-gw-signal', type=str, required=True, help='Path to the GW signal data frame file (HDF5 format)')
    parser.add_argument('--cleaned-data', type=str, required=True, help='Path to the cleaned GW data file (GWF format)')
    parser.add_argument('--injections-file', type=str, required=True, help='Path to the CSV file containing injection data')
    parser.add_argument('--channel-org', type=str, default='V1:Hrec_hoft_raw_20000Hz', help='Channel name for original data')
    parser.add_argument('--channel-DC', type=str, default='V1:Hrec_hoft_raw_20000Hz_DC', help='Channel name for cleaned data')
    parser.add_argument('--channel-inj', type=str, default='V1:DC_INJ', help='Channel name for injected data')
    parser.add_argument('--outdir', type=str, default='output', help='Directory for saving output files and plots')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')
    return parser.parse_args()

# Main Execution Block
if __name__ == "__main__":
    args = parse_cli_arguments()
    os.makedirs(args.outdir, exist_ok=True)  # Ensure output directory exists

    if args.verbose:
        print("Loading data...")

    # Load original, cleaned, and injection data
    if args.frame_org:
        original_data = load_time_series_from_h5(args.frame_org, args.channel_org)
        if args.verbose:
            print("Original data loaded.")
    cleaned_data = load_time_series_from_gwf(args.cleaned_data, args.channel_DC)
    if args.verbose:
        print("Cleaned data loaded.")

    injections_data = pd.read_csv(args.injections_file).iloc[1:-1]  # Exclude first and last entries
    if args.verbose:
        print("Injection data loaded.")

    crop_time = args.inj_gap / 2

    # Analysis logic: Matched filtering, PSD computation, SNR evaluation, and result plotting
    with h5py.File(args.frame_gw_signal, 'r') as file:
        gw_signal_data = file[args.channel_inj][:]
        start_time = file[args.channel_inj].attrs['x0']
        delta_time = file[args.channel_inj].attrs['dx']
        sample_rate = 1.0 / delta_time

        if args.verbose:
            print("GW signal data loaded and prepared for analysis.")

        # Convert GW signal data to TimeSeries and PyCBC types
        gw_signal_ts = TimeSeries(gw_signal_data, t0=start_time, sample_rate=sample_rate, name=args.channel_inj, unit="ct", channel=args.channel_inj).to_pycbc()
        filtered_gw_signal = highpass(gw_signal_ts, 15.0)
        filtered_cleaned_data = highpass(cleaned_data.to_pycbc(), 15.0)

        if args.verbose:
            print("Data filtered and ready for resampling and cropping.")

        # Resample and crop signals to match analysis duration
        resampled_gw_signal = resample_to_delta_t(filtered_gw_signal, 1/sample_rate).crop(crop_time, crop_time)
        resampled_cleaned_data = resample_to_delta_t(filtered_cleaned_data, 1/sample_rate).crop(crop_time, crop_time)

        if args.verbose:
            print("Signals resampled and cropped.")

        # Compute PSD for the cleaned data segment
        psd = resampled_cleaned_data.psd(8)
        
        psd = interpolate(psd, resampled_cleaned_data.delta_f)
        psd = inverse_spectrum_truncation(psd, int( 8 * resampled_cleaned_data.sample_rate), low_frequency_cutoff=20.0)

        print(len(psd))
        print(len(resampled_cleaned_data))
        if args.verbose:
            print("PSD computed for the cleaned data segment.")

        # Calculate SNR for the cleaned data using matched filtering
        snr_cleaned = matched_filter(resampled_gw_signal, resampled_cleaned_data, psd=psd, low_frequency_cutoff=25)
        snr_cleaned = snr_cleaned.crop(crop_time, crop_time)  # Crop edges to avoid edge effects

        if args.verbose:
            print("SNR calculated for the cleaned data.")

        peak_snr_values = []
        time_snr_peak = []
        num_inj = []
        
        n_inj = len(injections_data)
        for index, row in injections_data.iterrows():
            t0 = row['geocent_time']
            t1 = row['geocent_time'] + args.inj_gap
            start_index = int((t0 - start_time) * sample_rate)
            end_index = int((t1 - start_time) * sample_rate)

            if args.verbose:
                print(f"Processing injection {index}/{n_inj}...")

            # Error checking for indices
            if start_index < 0 or end_index > len(snr_cleaned):
                print(f"Error: Indices out of bounds for SNR array. Start: {start_index}, End: {end_index}")
                continue  # Skip this iteration and continue with the next

            snr_cleaned_select = snr_cleaned[start_index:end_index]
            if len(snr_cleaned_select) == 0:
                print(f"No data found in SNR array for indices {start_index} to {end_index}.")
                continue  # Skip this iteration and continue with the next

            peak_snr = abs(snr_cleaned_select).numpy().argmax()
            time = snr_cleaned_select.sample_times[peak_snr]
            peak_snr_values.append(abs(snr_cleaned_select[peak_snr]))
            time_snr_peak.append(time)
            num_inj.append(index)

            if args.verbose:
                print(f"Injection {index}: SNR = {abs(snr_cleaned_select[peak_snr])} at time {time}s")

        # Plot SNR over time
        plt.figure(figsize=(10, 6))
        plt.scatter(time_snr_peak, peak_snr_values, label='SNR After Cleaning')
        plt.xlabel('Time (s)')
        plt.ylabel('SNR')
        plt.title('SNR Improvement Post-Cleaning')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.outdir, 'snr_improvement_plot.png'))
        plt.show()

        if args.verbose:
            print("SNR improvement plot generated and saved.")

        # Save SNR data for further analysis
        snr_values = { 'Injection-num':num_inj,  'Time': time_snr_peak, 'SNR': peak_snr_values}
        snr_df = pd.DataFrame(snr_values)
        snr_df.to_csv(os.path.join(args.outdir, 'snr_values_post_clean.csv'), index=False)

        if args.verbose:
            print(f"SNR data saved. Analysis results are available in '{args.outdir}'.")
