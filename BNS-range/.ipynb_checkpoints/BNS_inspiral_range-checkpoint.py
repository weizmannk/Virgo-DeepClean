#!/usr/bin/env python

"""
---------------------------------------------------------------------------------------------------
ABOUT THE SCRIPT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository URL  : https://github.com/weizmannk/Virgo-DeepClean.git
Creation Date   : January 2024
Description     : The script is designed to calculate the Binary Neutron Star (BNS) inspiral range both before and after applying the DeepClean process to
                  gravitational wave data. It segments the data every 4096 seconds over two days, assessing the impact of noise reduction on the detection
                  sensitivity for BNS mergers, by assuming an SNR of 8 and a 1.4 solar mass binary system. The analysis is conducted over various frequency
                  bands.
Usage           : python BNS_inspiral_range.py --path-dir /path/to/data --outdir /path/to/output --fmin <freq min> --fmax <freq max>
Example         : python BNS_inspiral_range.py --path-dir /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/197-208_Hz/4096  --outdir ./SNR-Peak --fmin 197 --fmax 208
---------------------------------------------------------------------------------------------------
"""


#  python BNS_inspiral_range.py --path-dir-org /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/142-162_Hz/4096 --path-dir-dc /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/142-162_Hz/4096  --outdir ./data --fmin 142 --fmax 162


# python BNS_inspiral_range.py --path-dir-org /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/4096/layer0  --path-dir-dc /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/4096/layer14  --outdir ./BNS-range --fmin 15 --fmax 415


# python BNS_inspiral_range.py --path-dir-org /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/4096/layer0   --outdir ./BNS-range --fmin 15 --fmax 415  --path-dir-dc /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/4096/layer0  --layer layer0

# Script imports
import os
import glob
import argparse
import numpy as np
from tqdm.auto import tqdm
import h5py
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.astro import inspiral_range
import logging

# Initialize logging for informative output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculates BNS Inspiral Range Before and After the DeepClean Process."
    )
    # parser.add_argument('--path-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument(
        "--path-dir-org",
        type=str,
        required=True,
        help="Path to original data directory",
    )
    parser.add_argument(
        "--path-dir-dc",
        type=str,
        required=True,
        help="Path to DeepClean  data directory",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Output directory for plots"
    )
    parser.add_argument(
        "--channel-org",
        type=str,
        default="V1:Hrec_hoft_raw_20000Hz",
        help="Original channel name",
    )
    parser.add_argument(
        "--channel-dc",
        type=str,
        default="V1:Hrec_hoft_raw_20000Hz_DC",
        help="DeepClean channel name",
    )
    parser.add_argument(
        "--fmin", type=float, default=142, help="Minimum frequency for analysis"
    )
    parser.add_argument(
        "--fmax", type=float, default=162, help="Maximum frequency for analysis"
    )
    parser.add_argument("--layer", type=str, help="for the multi-training")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def load_time_series(file_path, channel, format="gwf"):
    """Load time series data from a file."""
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


def bns_inspiral_range(args):
    """Calculate and print the BNS inspiral range for the given data."""
    os.makedirs(args.outdir, exist_ok=True)

    # locs = glob.glob(os.path.join(args.path_dir, "Hrec*"), recursive=True)

    locs_dc = glob.glob(os.path.join(args.path_dir_dc, "Hrec*"), recursive=True)
    locs_org = glob.glob(os.path.join(args.path_dir_org, "Hrec*"), recursive=True)

    DC_bns_range = []
    Org_bns_range = []
    start_time = []
    duration_time = []

    #####################
    # dir_org = glob.glob(os.path.join("/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/4096/layer0", "Hrec*"), recursive=True)[0]
    ########################

    for loc_org, loc_dc in tqdm(zip(locs_org, locs_dc), desc="Processing locations"):
        frame_org_files = glob.glob(os.path.join(loc_org, "orig*h5"))
        frame_dc_files = glob.glob(os.path.join(loc_dc, "Hrec*.gwf"))

        # frame_org_files = glob.glob(os.path.join(dir_org, "original-1265127585-4096.h5"))

        for frame_dc, frame_org in zip(
            tqdm(sorted(frame_dc_files)), sorted(frame_org_files)
        ):
            start, duration = map(
                int, frame_org.split("original-")[1].split(".h5")[0].split("-")[:2]
            )

            print("=================================\n")
            print(frame_dc)
            print("++++++++++++++++++++++++++\n")
            print(frame_org)
            print("==================================\n")

            data_org = load_time_series(frame_org, args.channel_org, format="h5")
            data_dc = load_time_series(frame_dc, args.channel_dc, format="gwf")

            psd_org = data_org.psd(fftlength=8, overlap=0, method="median")
            psd_dc = data_dc.psd(fftlength=8, overlap=0, method="median")

            freq_org = psd_org.frequencies.value
            freq_dc = psd_dc.frequencies.value

            bns_range_org = inspiral_range(
                FrequencySeries(
                    psd_org.value, f0=freq_org[0], df=freq_org[1] - freq_org[0]
                ),
                snr=8,
                fmin=10,
                mass1=1.4,
                mass2=1.4,
            ).value

            bns_range_dc = inspiral_range(
                FrequencySeries(
                    psd_dc.value, f0=freq_dc[0], df=freq_dc[1] - freq_dc[0]
                ),
                snr=8,
                fmin=10,
                mass1=1.4,
                mass2=1.4,
            ).value

            start_time.append(start)
            duration_time.append(duration)
            Org_bns_range.append(bns_range_org)
            DC_bns_range.append(bns_range_dc)

            print(
                f"\n Start time : {start}, the BNS Inspiral range of ORG : {np.round(bns_range_org, 0)} Mpc  and DC :{np.round(bns_range_dc, 0)} Mpc\n"
            )
    # Save the results to a CSV file
    with open(
        os.path.join(
            args.outdir, f"BNS_inspiral_range_{int(args.fmin)}-{int(args.fmax)}_Hz.csv"
        ),
        "w",
    ) as f:
        f.write("Start Time,Duration,BNS Inspiral Range ORG,BNS Inspiral Range DC\n")
        for start, duration, org_range, dc_range in zip(
            start_time, duration_time, Org_bns_range, DC_bns_range
        ):
            f.write(f"{start},{duration},{org_range},{dc_range}\n")

    logging.info("BNS inspiral range analysis completed and results saved.")


if __name__ == "__main__":
    args = parse_arguments()
    bns_inspiral_range(args)
