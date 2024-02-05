#!/usr/bin/env python3

"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository      : https://github.com/weizmannk/Virgo-DeepClean.git
Created On      : January 2024
Description     : This script performs matched filtering analysis on gravitational wave (GW) data
                  before the application of the DeepClean algorithm. It is designed to load GW signal
                  data, compute Power Spectral Density (PSD), and process signal injections, offering
                  flexibility in specifying parameters through command-line arguments.
Usage           : python pre_cleanning_match_filter_snr_freq.py --inj-gap <int> --frame-org <path> 
                  --frame-gw-signal <path> --injections-file <path> --outdir <path> [--verbose]
                   
Example         : python pre_cleanning_match_filter_snr_freq.py --inj-gap 32 --frame-org original-1265127585-4096.h5 --channel-org V1:Hrec_hoft_raw_20000Hz --frame-inj injected_original-1265127585-4096.h5 --frame-gw-signal V1_BBH_inj_1265127585.0_1265131681.0.hdf5  --channel-inj V1:DC_INJ --injections-file INJ-1265127585-4096.csv --fmin 145 --fmax 155
---------------------------------------------------------------------------------------------------
"""

# Import Statements
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

def compute_snr(template, data, sample_rate, crop_time, low_frequency_cutoff=20):
    """Compute the SNR of a GW signal using matched filtering."""
    
    filtered_data = highpass(data, 15)
    filtered_template = highpass(template, 15)
    resampled_data = resample_to_delta_t(filtered_data, 1 / sample_rate).crop(crop_time, crop_time)
    resampled_template = resample_to_delta_t(filtered_template, 1 / sample_rate).crop(crop_time, crop_time)
    time_psd = sample_rate / 8
    psd = resampled_data.psd(time_psd)
    psd = interpolate(psd, resampled_data.delta_f)
    psd = inverse_spectrum_truncation(psd, int(time_psd * sample_rate), low_frequency_cutoff=20)
    snr = matched_filter(resampled_template, resampled_data, psd=psd, low_frequency_cutoff=low_frequency_cutoff)
    snr_cropped = snr.crop(crop_time, crop_time)
    return snr_cropped

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
    parser.add_argument('--fmin', type=float, required=True, help='Minimum frequency for classification (Hz)')
    parser.add_argument('--fmax', type=float, required=True, help='Maximum frequency for classification (Hz)')

    return parser.parse_args()



# Main Execution Block
if __name__ == "__main__":
    args = parse_cli_arguments()
    # Define the output directory for plots and ensure it exists
    os.makedirs(args.outdir, exist_ok=True)

    
    # Load original data if provided
    original_data = load_time_series(args.frame_org, args.channel_org, format='h5') if args.frame_org else None
    
    # # Load injections data ....
    data_inj  = load_time_series(args.frame_inj, args.channel_org, format='h5')
    
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
        data = data_inj.to_pycbc()

        # Calculate SNR for the cleaned data using matched filtering
        snr = compute_snr(template=template, data=data, sample_rate=sample_rate, crop_time=crop_time, low_frequency_cutoff=25)

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
                end_index = min(len(snr), int((t1 - start_time) * sample_rate))

                if start_index >= end_index or start_index >= len(snr):
                    if args.verbose:
                        print(f"Invalid index range for event {index}: start_index={start_index}, end_index={end_index}, len(snr)={len(snr_cleaned)}")
                    continue

                snr_event = snr[start_index:end_index]

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
        with open(os.path.join(args.outdir, f'Pre-clean_snr_peaks_all_classifications_{fmin}-{fmax}.csv'), 'w') as f:
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
        plt.title(f'SNR Peaks by Classification Pre-Cleaning {fmin}-{fmax}')
        plt.legend()
        plt.grid(True)

        # Save the plot to a file
        plt.savefig(os.path.join(args.outdir, f'Pre_clean_snr_peaks_by_classification_{fmin}-{fmax}.png'))

        # Optionally display the plot
        plt.show()

        if args.verbose:
            print("SNR peaks by classification plot generated and saved.")
