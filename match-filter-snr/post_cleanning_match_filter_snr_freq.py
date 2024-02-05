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
                  to evaluate signal-to-noise ratio (SNR) improvements post-cleaning using the
                  DeepClean algorithm. It processes cleaned GW signal data from GWF files, executes
                  matched filtering, computes Power Spectral Density (PSD), and evaluates SNR on
                  injected signals. The script is designed for scientific workflows to verify
                  GW signal integrity after cleaning.
Usage           : python post_cleanning_match_filter_snr_freq.py [options]
Options         : --inj-gap <int>, --frame-org <path>, --channel-org <name>, --frame-gw-signal <path>,
                  --cleaned-data <path>, --channel-DC <name>, --channel-inj <name>, --injections-file <path>,
                  --outdir <path>, --verbose

Example         : python post_cleanning_match_filter_snr_freq.py --inj-gap 32 --channel-org V1:Hrec_hoft_raw_20000Hz --frame-gw-signal V1_BBH_inj_1265127585.0_1265131681.0.hdf5 --cleaned-data injected_clean_Hrec-HOFT-1265127585-4096.gwf --channel-DC V1:Hrec_hoft_raw_20000Hz_DC --channel-inj V1:DC_INJ --injections-file INJ-1265127585-4096.csv --verbose
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
import numpy as np

# Constants
G = 6.6743e-11  # Newton's gravitational constant (m^3/kg/s^2)
c = 299792458.0  # Speed of light in vacuum (m/s)
Sun_Mass = 1.988409870698051e+30  # Mass of the sun (kg)


# Function Definitions
def gw_frequency_at_isco(mass):
    """Calculate GW frequency at ISCO for a given total mass."""
    return c**3 / (6**1.5 * np.pi * G * mass * Sun_Mass)

def get_M_for_flso (fisco):

    mass = c**3 / (6**1.5 * np.pi * G * fisco * Sun_Mass)
    return mass

# Function Definitions
def gw_frequency_at_isco(mass):
    """Calculate GW frequency at ISCO for a given total mass."""
    return c**3 / (6**1.5 * np.pi * G * mass * Sun_Mass)

def merger_frequency_approx(mass):
    """Estimate GW frequency at merger using the 15 kHz/M approximation."""
    return 15 * 1e3 / mass

def classify_events(injections_data, fmin, fmax):
    """Classify events based on ISCO and merger frequencies."""
    classifications = {'isco': [], 'merger': [], 'below_fmin': [], 'above_fmax': []}
    for index, row in injections_data.iterrows():
        m1, m2 = row['mass_1'], row['mass_2']
        total_mass = m1 + m2
        fisco = gw_frequency_at_isco(total_mass)
        fmerger = merger_frequency_approx(total_mass)
        #print("merger :", fmerger , " isco:" ,  fisco)         
        
        if fmin <= fisco <= fmax:
            classifications['isco'].append(index)
            print("fisco : ",fisco, "index :", index)
        elif fmin <= fmerger <= fmax:
            classifications['merger'].append(index)
            print("fmerger : ", fmerger,  "index :", index)
        elif fisco < fmin:
            classifications['below_fmin'].append(index)
        elif fmerger > fmax:
            classifications['above_fmax'].append(index)
    return classifications

def load_time_series(file_path, channel, format='gwf'):
    """Load a TimeSeries object from a file."""
    if format == 'gwf':
        return TimeSeries.read(file_path, channel)
    elif format == 'h5':
        with h5py.File(file_path, 'r') as file:
            data = file[channel][:]
            t0 = file[channel].attrs['t0']
            fs = file[channel].attrs['sample_rate']
        return TimeSeries(data, t0=t0, sample_rate=fs, name=channel, unit="ct", channel=channel)


def compute_snr(template, data, sample_rate,  crop_time , low_frequency_cutoff=20):
    """
    Compute the SNR of a gravitational wave signal using matched filtering, including preprocessing steps.

    Parameters:
    data (TimeSeries): The time series data of the cleaned GW signal.
    template (TimeSeries): The template waveform for matched filtering.
    sample_rate (float): The sample rate for resampling the data and template.
    inj_gap (float): The time gap between injections, used to determine crop time.
    low_frequency_cutoff (float): The low frequency cutoff for filtering the template and the data.

    Returns:
    TimeSeries: The SNR time series, cropped to remove edge effects.
    """

    # Highpass filter the data and template
    filtered_data = highpass(data, 15)
    filtered_template = highpass(template, 15)

    # Resample the filtered data and template
    resampled_data = resample_to_delta_t(filtered_data, 1 / sample_rate).crop(crop_time, crop_time)
    resampled_template = resample_to_delta_t(filtered_template, 1 / sample_rate).crop(crop_time, crop_time)

    # Compute PSD for the cleaned data segment
    time_psd = sample_rate/8
    psd = resampled_data.psd(time_psd)  # Use 4 seconds of data for PSD estimation
    psd = interpolate(psd, resampled_data.delta_f)
    psd = inverse_spectrum_truncation(psd, int(time_psd * sample_rate), low_frequency_cutoff=20)

    # Perform matched filtering
    snr = matched_filter(resampled_template, resampled_data, psd=psd, low_frequency_cutoff=low_frequency_cutoff)

    # Crop the SNR time series to remove edge effects
    snr_cropped = snr.crop(crop_time, crop_time)

    return snr_cropped

def parse_cli_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Matched filtering on GW data for SNR analysis.")
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
    parser.add_argument('--fmin', type=float, required=True, help='Minimum frequency for classification (Hz)')
    parser.add_argument('--fmax', type=float, required=True, help='Maximum frequency for classification (Hz)')
    return parser.parse_args()

# Main Execution Block
if __name__ == "__main__":
    args = parse_cli_arguments()
    os.makedirs(args.outdir, exist_ok=True)

    # Data loading and preprocessing steps...
    cleaned_data = load_time_series(args.cleaned_data, args.channel_DC, format='gwf')
    injections_data = pd.read_csv(args.injections_file)[1:-1]
    print(len(injections_data))
    
    crop_time = args.inj_gap / 2

    # Analysis logic: Matched filtering, PSD computation, SNR evaluation, and result plotting
    with h5py.File(args.frame_gw_signal, 'r') as file:
        gw_signal_data = file[args.channel_inj][:]
        start_time = file[args.channel_inj].attrs['x0']
        delta_time = file[args.channel_inj].attrs['dx']
        sample_rate = 1.0 / delta_time

        if args.verbose:
            print("GW signal data loaded and prepared for analysis.")

        # Convert GW signal template to TimeSeries and PyCBC types
        gw_signal_ts = TimeSeries(gw_signal_data, t0=start_time, sample_rate=sample_rate, name=args.channel_inj, unit="ct", channel=args.channel_inj)

        template = gw_signal_ts.to_pycbc()
        data = cleaned_data.to_pycbc()

        # Calculate SNR for the cleaned data using matched filtering
        snr_cleaned = compute_snr(template=template, data=data, sample_rate=sample_rate,  crop_time = crop_time, low_frequency_cutoff=25)

        # Classify events based on their frequencies
        classifications = classify_events(injections_data = injections_data, fmin=args.fmin, fmax=args.fmax)
    
        # Initialize a dictionary to hold SNR results for analysis
        snr_results = {}

        for classification, indices in classifications.items():
            snr_peaks = []
            for index in indices:
                event = injections_data.loc[index]
                t0 = event['geocent_time']
                t1 = t0 + args.inj_gap
                start_index = max(0, int((t0 - start_time) * sample_rate))
                end_index = min(len(snr_cleaned), int((t1 - start_time) * sample_rate))

                if start_index >= end_index or start_index >= len(snr_cleaned):
                    if args.verbose:
                        print(f"Invalid index range for event {index}: start_index={start_index}, end_index={end_index}, len(snr_cleaned)={len(snr_cleaned)}")
                    continue

                snr_event = snr_cleaned[start_index:end_index]

                if len(snr_event) == 0:
                    if args.verbose:
                        print(f"No SNR data for event {index} in range {start_index}-{end_index}.")
                    continue
                
                
                peak_snr_index = abs(snr_event).numpy().argmax()
                peak_snr_value = abs(snr_event[peak_snr_index])
                peak_snr_time = snr_event.sample_times[peak_snr_index]

                snr_peaks.append((index, peak_snr_value, peak_snr_time))
            

            snr_results[classification] = snr_peaks
            
        # Open a single file for writing the results of all classifications
        with open(os.path.join(args.outdir, f'Post-clean_snr_peaks_all_classifications_{fmin}-{fmax}.csv'), 'w') as f:
            # Write a header row to the file
            f.write("Classification,Index,Peak_SNR,Peak_SNR_Time\n")

            # Iterate through each classification and write its peaks
            for classification, peaks in snr_results.items():
                for index, peak, time in peaks:
                    f.write(f"{classification},{index},{peak},{time}\n")


        # Initialize a figure for plotting
        plt.figure(figsize=(10, 6))

        # Loop through each classification to plot its SNR peaks
        for classification, peaks in snr_results.items():
            # Extract event indices and corresponding peak SNR values
            indices = [peak[0] for peak in peaks]  # Event indices might not be necessary for the plot
            snr_values = [peak[1] for peak in peaks]  # SNR peak values

            # Since 'indices' are not directly useful for plotting against SNR values, you might want to use the event times
            # If 'geocent_time' is available and preferable, consider replacing 'indices' with a list of 'geocent_times'
            # Assuming 'geocent_times' is a list of corresponding event times for the 'indices'
            geocent_times = [injections_data.loc[index]['geocent_time'] for index in indices]

            # Plot SNR values for the current classification
            plt.scatter(geocent_times, snr_values, label=classification, alpha=0.7)

        plt.xlabel('Time (s)')
        plt.ylabel('SNR')
        plt.title(f'SNR Peaks by Classification Post-Cleaning for {fmin}-{fmax}')
        plt.legend()
        plt.grid(True)

        # Save the plot to a file
        plt.savefig(os.path.join(args.outdir, f'Post_Clean_snr_peaks_by_classification_{fmin}-{fmax}.png'))

        # Optionally display the plot
        plt.show()

        if args.verbose:
            print("SNR peaks by classification plot generated and saved.")
