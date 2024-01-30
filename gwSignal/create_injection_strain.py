#!/usr/bin/env python3

"""
ABOUT THE SCRIPT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository URL  : https://github.com/weizmannk/ObservingScenariosInsights.git
Creation Date   : January 2024
Description     : This script creates injections from a given prior and generates time-domain waveform strain
                  for gravitational wave data analysis. It supports multiple interferometers and allows for
                  various waveform and injection configurations.
                  Originally written by Colm Talbot and  Muhammed Saleem
Example         : ./create_injection_strain.py --interferometers V1  --inj-gap 32 --prior-file prior.prior --start-time 1265127585 --duration 4096 --frame-duration 4096 --sampling-frequency 4096 --inject --outdir injection_Dir_98_108_Hz 

---------------------------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import bilby
import gwpy
import lalsimulation as lalsim
import lal
from argparse import ArgumentParser
from scipy.signal.windows import tukey
from tqdm.auto import trange
from bilby.core.prior import PriorDict, Uniform
from bilby.core.utils import check_directory_exists_and_if_not_mkdir, logger
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.detector import InterferometerList, PowerSpectralDensity
from bilby.gw.prior import BBHPriorDict
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.waveform_generator import WaveformGenerator
from bilby_pipe.utils import convert_string_to_dict
from gwpy.timeseries import TimeSeries
from gwpy.detector import Channel

# Parse command-line arguments
parser = ArgumentParser(description="Create injections from prior and generate time-domain waveform strain")
parser.add_argument("--n-injections", type=int, default=32, help="Number of injections")
parser.add_argument("--inj-gap", default=None, type=float, help="Gap between injections in seconds. Overrides --n-injections if provided.")
parser.add_argument("--prior-file", type=str, required=True, help="Bilby prior file")
parser.add_argument("--start-time", type=float, required=True, help="Start time for injections.")
parser.add_argument("--duration", type=int, default=4096, help="Duration of the data segment in seconds.")
parser.add_argument("--frame-duration", type=int, default=4096, help="Duration of each data frame in seconds.")
parser.add_argument("--outdir", default=".", type=str, help="Path to the output directory")
parser.add_argument("--sampling-frequency", type=float, default=4096, help="Sampling frequency in Hz")
parser.add_argument("--channel-name", type=str, default="DC_INJ", help="Channel name for the injected strain data")
parser.add_argument("--format", type=str, default="hdf5", help="Format to save the injected data (hdf5, txt, etc.)")
parser.add_argument("--interferometers", default=["H1", "L1"], nargs="*", help="List of interferometers for which to generate data")
parser.add_argument("--inject", action="store_true", help="Flag to perform injection")
parser.add_argument("--psd-dict", type=str, default="default", help="PSD dictionary in bilby_pipe format for noise generation")
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.outdir, exist_ok=True)

# Prepare filenames for injection details
injection_file_json = os.path.join(args.outdir, f'INJ-{int(args.start_time)}-{int(args.duration)}.json')
injection_file_csv = os.path.join(args.outdir, f'INJ-{int(args.start_time)}-{int(args.duration)}.csv')

# Load priors from the specified file
prior = BBHPriorDict(args.prior_file)
end_time = args.start_time + args.duration

# Generate injection times
if args.inj_gap is None:
    # Randomly space injections within the given duration
    prior['geocent_time'] = Uniform(args.start_time, end_time)
    samples = prior.sample(args.n_injections)
    samples['geocent_time'] = np.sort(samples['geocent_time'])
else:
    # Generate equally spaced injections
    times = np.arange(args.start_time, end_time, args.inj_gap)
    samples = prior.sample(len(times))
    samples['geocent_time'] = np.sort(times)

# Save injections to files
injections = pd.DataFrame(samples)
injections.to_json(injection_file_json)
injections.to_csv(injection_file_csv)
logger.info("Injections saved as JSON and CSV.")

def configure_ifos(ifos, args, psd_dict):
    for ifo in ifos:
        ifo.minimum_frequency = 10
        if ifo.name in psd_dict and psd_dict[ifo.name]:
            psd_file = psd_dict[ifo.name]
            if "asd" in psd_file:
                method = PowerSpectralDensity.from_amplitude_spectral_density_file
            elif "psd" in psd_file:
                method = PowerSpectralDensity.from_power_spectral_density_file
            else:
                method = PowerSpectralDensity.from_power_spectral_density_file
            ifo.power_spectral_density = method(psd_file)
        else:
            logger.warning(f"No PSD or ASD provided for {ifo.name}, using default.")

def td_waveform(params, args):
    params_new, _ = convert_to_lal_binary_black_hole_parameters(params)
    chirp_time = lalsim.SimInspiralChirpTimeBound(
        10,
        params_new["mass_1"] * lal.MSUN_SI,
        params_new["mass_2"] * lal.MSUN_SI,
        params_new["a_1"] * np.cos(params_new["tilt_1"]),
        params_new["a_2"] * np.cos(params_new["tilt_2"])
    )
    chirp_time = max(2 ** (int(np.log2(chirp_time)) + 1), 4)
    wfg = WaveformGenerator(
        duration=chirp_time,
        sampling_frequency=args.sampling_frequency,
        start_time=params["geocent_time"] + 0.2 - chirp_time,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={"minimum_frequency": 10.0}
    )

    wf_pols = wfg.time_domain_strain(params)
    window = tukey(len(wfg.time_array), alpha=0.2 / chirp_time)
    wf_pols["plus"] *= window
    wf_pols["cross"] *= window
    wf_pols["plus"] = np.roll(wf_pols["plus"], -int(args.sampling_frequency * 0.2))
    wf_pols["cross"] = np.roll(wf_pols["cross"], -int(args.sampling_frequency * 0.2))
    return wf_pols, wfg.time_array

def do_injections(ifos, injections, channels, strain, args):
    strain_with_injs = strain.copy()
    for inj_no, parameters in injections.iterrows():
        wf_pols, times = td_waveform(parameters, args)
        for ifo in ifos:
            ifo_strain = strain_with_injs[ifo.name]
            time_delay = ifo.time_delay_from_geocenter(parameters["ra"], parameters["dec"], parameters["geocent_time"])
            times_delayed = times + time_delay
            signal = ifo.get_detector_response(wf_pols, parameters, times_delayed)
            inj_strain = TimeSeries(signal, times=times_delayed)
            ifo_strain = ifo_strain.add(inj_strain)
            strain_with_injs[ifo.name] = ifo_strain
    return strain_with_injs

def save_strain(strain_with_injs, start_time, frame_end_time, args):
    for ifo_name, ifo_strain in strain_with_injs.items():
        file_name = f"{args.outdir}/{ifo_name}_BBH_inj_{start_time}_{frame_end_time}.{args.format}"
        logger.info(f"Saving {file_name}")
        ifo_strain.write(file_name, format=args.format)

def main(args, injections):
    check_directory_exists_and_if_not_mkdir(args.outdir)

    psd_dict = convert_string_to_dict(args.psd_dict) if args.psd_dict != "default" else {}
    ifos = InterferometerList(args.interferometers)
    configure_ifos(ifos, args, psd_dict)

    strain = {}
    start_time = args.start_time
    while start_time < end_time:
        frame_end_time = min(start_time + args.frame_duration, end_time)
        ifos.set_strain_data_from_zero_noise(args.sampling_frequency, args.frame_duration, start_time)
        strain_with_injs = do_injections(ifos, injections, {ifo.name: f"{ifo.name}:{args.channel_name}" for ifo in ifos}, strain, args)
        save_strain(strain_with_injs, start_time, frame_end_time, args)
        start_time += args.frame_duration

if __name__ == "__main__":
    main(args, injections)
