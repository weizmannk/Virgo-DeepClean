#!/usr/bin/env python

"""
---------------------------------------------------------------------------------------------------
ASD & ASD Ratio Plotter for DeepClean Comparison
---------------------------------------------------------------------------------------------------
Author         : Ramodgwendé Weizmann KIENDREBEOGO
Contact        : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository     : https://github.com/weizmannk/Virgo-DeepClean.git
Created        : January 2025

Description    : Generate publication-ready plots to compare Amplitude Spectral Density (ASD) and
                 ASD ratios for original, DeepClean-single, and DeepClean-multi training strategies
                 on gravitational wave data. Highlights the impact of sequential (layered) versus
                 simultaneous witness training for noise removal.
---------------------------------------------------------------------------------------------------
Usage Examples:

    # To process a single layer (e.g. layer 1)
    python asd-asd-ratio-seq-single.py --outdir ./Plots_test --fmin 142 --fmax 162 \
        --channel-dc-sing V1:Hrec_hoft_raw_20000Hz_DC --layer-dc-seq 1

    # To process a range of layers (from layer 0 up to but not including layer 8)
    python asd-asd-ratio-seq-single.py --outdir ./Plots_test --fmin 142 --fmax 162 \
        --channel-dc-sing V1:Hrec_hoft_raw_20000Hz_DC  --layer-min 0 --layer-max 8
---------------------------------------------------------------------------------------------------

"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import h5py
from gwpy.timeseries import TimeSeries
import logging


#layer_number = 1


plt.switch_backend('agg')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')



COLOR_ORG = "#28B463"      # Vivid green
COLOR_MULTI = "#9400D3"    # Violet (DeepClean-Multi/Layered)
COLOR_SINGLE = "#E74C3C"   # Strong red (DeepClean-Single)
COLOR_RATIO_MULTI = "#1F77B4"  # Blue
COLOR_RATIO_SINGLE = "#222222" # Black/Gray
COLOR_RATIO_FINAL = "#FFA500"  # Orange


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate publication-quality ASD & ASD ratio plots for DeepClean training comparison.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--channel-org', type=str, default='V1:Hrec_hoft_raw_20000Hz', help='Original data channel')
    parser.add_argument('--channel-dc-sing', type=str, default='V1:Hrec_hoft_raw_20000Hz_DC', help='DeepClean data channel (single)')
    parser.add_argument(
        '--layer-dc-seq', type=int,
        help='Process a specific layer  (e.g., 1, overrides layer-min/layer-max), (multi/seq)'
    )
    parser.add_argument(
        '--layer-min', type=int, default=0,
        help='First value of layer (included), (multi/seq)'
    )
    parser.add_argument(
        '--layer-max', type=int, default=None,
        help='Last value of layer (excluded), (multi/seq)'
    )

    parser.add_argument('--fmin', type=float, default=142, help='Minimum frequency [Hz]')
    parser.add_argument('--fmax', type=float, default=162, help='Maximum frequency [Hz]')
    parser.add_argument(
                        '--witness-channels-dir', type=str,
                        default='./witnesses-sequential',
                        help='Directory containing witness channels (.ini files) for sequential DeepClean'
)

    return parser.parse_args()


def plot_asd_and_ratio(data_org, data_dc_seq, data_dc_sing, figname, start, end, fmin, fmax):
    """Plot ASD and ASD ratio for original, DeepClean-single, and DeepClean-multi data."""
    asd_org = data_org.asd(fftlength=32, overlap=0, method='median').crop(fmin, fmax)
    asd_dc_seq = data_dc_seq.asd(fftlength=32, overlap=0, method='median').crop(fmin, fmax)
    asd_dc_sing = data_dc_sing.asd(fftlength=32, overlap=0, method='median').crop(fmin, fmax)
    ratio_multi = asd_dc_seq / asd_org
    ratio_single = asd_dc_sing / asd_org
    ratio_gain = ratio_single / ratio_multi

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    axs[0].loglog(asd_org, label='V1:ORG', color=COLOR_ORG, linewidth=2.2)
    axs[0].loglog(asd_dc_seq, label='V1:DC Multi-layer', color=COLOR_MULTI, linewidth=2.2)
    axs[0].loglog(asd_dc_sing, label='V1:DC Single-pass', color=COLOR_SINGLE, linewidth=2.2)
    axs[0].set_ylabel(r'ASD [Hz$^{-1/2}$]', fontsize=13)
    axs[0].set_xlim(fmin, fmax)
    axs[0].set_ylim(1e-24, 1e-20)
    axs[0].legend(loc='best')
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    axs[1].loglog(ratio_multi, label='Multi-layer / ORG', color=COLOR_RATIO_MULTI, linewidth=2)
    axs[1].loglog(ratio_single, label='Single-pass / ORG', color=COLOR_RATIO_SINGLE, linewidth=2)
    axs[1].set_xlabel('Frequency [Hz]', fontsize=13)
    axs[1].set_ylabel('ASD Ratio', fontsize=13)
    axs[1].set_xlim(fmin, fmax)
    axs[1].set_ylim(0.1, 1.5)
    axs[1].legend(loc='best')
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    fig.suptitle(f'ASD and ASD Ratio from {start} to {end}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(figname, dpi=300)
    logging.info(f'Plot saved to {figname}')
    plt.close(fig)
    
    return ratio_multi, ratio_single, ratio_gain

def generate_asd_plots(args, layer_str):
    
    try:
        os.makedirs(args.outdir, exist_ok=True)
        layer_number = layer_str

        # Replace with CLI/config as needed
        frame_dc_seq = f"/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/rework_paper/Check/linear_subtraction/4096_sequential/{layer_number}/Hrec-HOFT-1265127585-100000/Hrec-HOFT-1265127585-4096_{layer_number}.gwf"
        frame_dc_sing = "/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/142-162_Hz/4096/Hrec-HOFT-1265127585-50000/Hrec-HOFT-1265127585-4096.gwf"
        frame_org = "/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/rework_paper/Check/linear_subtraction/4096_sequential/layer0/Hrec-HOFT-1265127585-100000/original-1265127585-4096.h5"
        start, duration = map(int, (frame_org.split("original-")[1].split(".h5")[0].split("-")[:2]))
        data_org = timeSeries_from_h5(frame_org, args.channel_org)
        
        if layer_number== "layer121":
            data_dc_seq = TimeSeries.read(frame_dc_seq,  "V1:Hrec_hoft_raw_20000Hz_DC")
        else:
            data_dc_seq = TimeSeries.read(frame_dc_seq, layer_number)
        
        data_dc_sing = TimeSeries.read(frame_dc_sing, args.channel_dc_sing)
        figname = os.path.join(args.outdir, f"asd_comparison_{start}-{duration}_{args.fmin}-{args.fmax}_Hz_{layer_number}.pdf")
        ratio_multi, ratio_single, ratio_gain = plot_asd_and_ratio(
            data_org, data_dc_seq, data_dc_sing, figname, start, start + duration, args.fmin, args.fmax
        )
        return ratio_multi, ratio_single, ratio_gain

    except FileNotFoundError as e:
        logging.warning(f"File not found for layer {layer_str}: {e}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error processing layer {layer_str}: {e}")
        return None, None, None

def timeSeries_from_h5(filename, channel='V1:Hrec_hoft_raw_20000Hz'):
    """Read .h5 output files produced by dc-prod-clean into a GWpy TimeSeries."""
    with h5py.File(filename, 'r') as f:
        data = f[channel][:]
        t0 = f[channel].attrs['t0']
        fs = f[channel].attrs['sample_rate']
    return TimeSeries(data, t0=t0, sample_rate=fs, name=channel, unit="ct", channel=channel)


def ratio_values(args, layer_str):
    """
    Read channel information from a .ini file for a specific layer.
    Returns (layer_number, channel_name).
    """
    layer_ix = layer_str
    if layer_ix == "layer121":
        channel_name = "V1:Hrec_hoft_raw_20000Hz_DC" 
        
    else:   
        input_filename = f'{args.witness_channels_dir}/{layer_ix}.ini'
        with open(input_filename, 'r') as infile:
            lines = [l.strip() for l in infile if l.strip()]

            channel_name = lines[-1]
    
    layer_number = layer_ix
    print("layer :", layer_ix,  "\n channel name :", channel_name)
    return layer_ix, channel_name


def main():
    args = parse_arguments()
    
    layers, channels = [], []
    max_ratio_multi_all, max_ratio_single_all, max_ratio_gain_all = [], [], [] 
    
    if args.layer_dc_seq is not None:
        indices = [args.layer_dc_seq]
        
    elif args.layer_max is not None:
        indices = list(range(args.layer_min, args.layer_max))
    else:
        parser.error("You must provide either --layer-dc-seq (single layer) or --layer-max (for a range)")

    for num in indices:
        layer_str = f"layer{num}"
    
        layer_number, channel_name = ratio_values(args, layer_str)
            
        ratio_multi, ratio_single, ratio_gain = generate_asd_plots(args, layer_str)
        
        if any(x is None for x in (ratio_multi, ratio_single, ratio_gain)):
            print(f"Skipping layer {layer_str} due to missing files.")
            continue

        # Optionally round maxima for reporting
        max_ratio_multi =  np.max(ratio_multi.value) #np.round(np.max(ratio_multi.value), 12)
        max_ratio_single = np.max(ratio_single.value) #np.round(np.max(ratio_single.value), 12)
        max_ratio_gain = np.max(ratio_gain.value)   # np.round(np.max(ratio_gain.value), 12)

        logging.info(f"Max Ratio (Multi): {max_ratio_multi}")
        logging.info(f"Max Ratio (Single): {max_ratio_single}")
        logging.info(f"Max Gain (Single/Multi): {max_ratio_gain}")

        layers.append(layer_number)
        max_ratio_multi_all.append(max_ratio_multi)
        max_ratio_single_all.append(max_ratio_single)
        max_ratio_gain_all.append(max_ratio_gain)
        channels.append(channel_name)


    tab = Table(
        [
            layers, 
            max_ratio_multi_all, 
            max_ratio_single_all, 
            max_ratio_gain_all, 
            channels
        ],
        names=[
            "layer_number",
            "max_ratio_multi_ORG",
            "max_ratio_single_ORG",
            "max_ratio_single_multi",
            "witness_channels"
        ]
    )

    print(tab)

if __name__ == "__main__":
    main()


