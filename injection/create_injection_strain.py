#!/usr/bin/env python3

## python create_injection_strain.py --interferometers V1  --inj-gap 32 --prior-file  ./priors/BBH_V1_15-420_Hz.prior  --start-time 1265127585 --duration 4096 --frame-duration 4096 --sampling-frequency 4096 --inject --outdir ./15-415_Hz/GWsignals


# python create_injection_strain.py --interferometers V1  --inj-gap 32 --prior-file  ./priors/BBH_V1_142-165_Hz_5_Mpc.prior  --start-time 1265127585 --duration 4096 --frame-duration 4096 --sampling-frequency 4096 --inject --outdir ./142-162_5_Mpc


# python create_injection_strain.py --interferometers V1  --inj-gap 32 --prior-file  ./priors/BBH_V1_15-420_Hz.prior  --start-time 1265127585 --duration 4096 --frame-duration 4096 --sampling-frequency 4096 --inject --outdir  ./rework_15_415_Hz/injSignal


#
"""
ABOUT THE SCRIPT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository URL  : https://github.com/weizmannk/Virgo-DeepClean.git
Creation Date   : January 2024
Description     : This script creates injections from a given prior and generates time-domain waveform strain
                  for gravitational wave data analysis. It supports multiple interferometers and allows for
                  various waveform and injection configurations.
                  Originally written by Colm Talbot and  Muhammed Saleem
Example         : python create_injection_strain.py --interferometers V1  --inj-gap 32 --prior-file prior.prior --start-time 1265127585 --duration 4096 --frame-duration 4096 --sampling-frequency 4096 --inject --outdir injection_Dir_98_108_Hz

---------------------------------------------------------------------------------------------------
"""

import numpy as np
import bilby
from bilby.core.prior import PriorDict, Uniform
import glob
import os
import pandas as pd
from argparse import ArgumentParser

from scipy.signal.windows import tukey
from tqdm.auto import trange

import gwpy
import gwpy.timeseries as ts
from gwpy.detector import Channel

import lalsimulation as lalsim
import lal

from bilby.core.utils import check_directory_exists_and_if_not_mkdir, logger

from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.detector import InterferometerList, PowerSpectralDensity
from bilby.gw.prior import BBHPriorDict
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.waveform_generator import WaveformGenerator
from bilby_pipe.utils import convert_string_to_dict

# Parse command-line arguments
parser = ArgumentParser(
    description="Create injections from prior and generate time-domain waveform strain"
)
parser.add_argument("--n-injections", type=int, default=32, help="Number of injections")
parser.add_argument(
    "--inj-gap",
    default=None,
    type=float,
    help="Gap between injections in seconds. Overrides --n-injections if provided.",
)
parser.add_argument("--prior-file", type=str, required=True, help="Bilby prior file")
parser.add_argument(
    "--start-time", type=float, required=True, help="Start time for injections."
)
parser.add_argument(
    "--duration",
    type=int,
    default=4096,
    help="Duration of the data segment in seconds.",
)
parser.add_argument(
    "--frame-duration",
    type=int,
    default=4096,
    help="Duration of each data frame in seconds.",
)
parser.add_argument(
    "--outdir", default=".", type=str, help="Path to the output directory"
)
parser.add_argument(
    "--sampling-frequency", type=float, default=4096, help="Sampling frequency in Hz"
)
parser.add_argument(
    "--channel-name",
    type=str,
    default="DC_INJ",
    help="Channel name for the injected strain data",
)
parser.add_argument(
    "--format",
    type=str,
    default="hdf5",
    help="Format to save the injected data (hdf5, txt, etc.)",
)
parser.add_argument(
    "--interferometers",
    default=["H1", "L1"],
    nargs="*",
    help="List of interferometers for which to generate data",
)
parser.add_argument("--inject", action="store_true", help="Flag to perform injection")
parser.add_argument(
    "--psd-dict",
    type=str,
    default="default",
    help="PSD dictionary in bilby_pipe format for noise generation",
)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

injection_file_json = "INJ-{:d}-{:d}.json".format(
    int(args.start_time), int(args.duration)
)
injection_file_json = os.path.join(args.outdir, injection_file_json)

injection_file_csv = "INJ-{:d}-{:d}.csv".format(
    int(args.start_time), int(args.duration)
)
injection_file_csv = os.path.join(args.outdir, injection_file_csv)

# prior = PriorDict(filename=args.prior_file)
prior = BBHPriorDict(args.prior_file)

end_time = args.start_time + args.duration


if args.inj_gap == None:
    # randomly space <args.n_injections> injections
    prior["geocent_time"] = Uniform(args.start_time, end_time)
    samples = prior.sample(args.n_injections)
    samples["geocent_time"] = np.sort(samples["geocent_time"])

else:
    # equally spaced injections with <> seconds between (the merger epochs)
    # of the  two nearby injections

    times = np.arange(args.start_time, end_time, args.inj_gap)
    samples = prior.sample(len(times))
    samples["geocent_time"] = np.sort(times)


injections = pd.DataFrame(samples)
injections.to_json(injection_file_json)
injections.to_csv(injection_file_csv)

logger.info(f"Writing injections as json and csv . . ")

################################################
def td_waveform(params, args):
    params_new, _ = convert_to_lal_binary_black_hole_parameters(params)
    chirp_time = lalsim.SimInspiralChirpTimeBound(
        10,
        params_new["mass_1"] * lal.MSUN_SI,
        params_new["mass_2"] * lal.MSUN_SI,
        params_new["a_1"] * np.cos(params_new["tilt_1"]),
        params_new["a_2"] * np.cos(params_new["tilt_2"]),
    )
    chirp_time = max(2 ** (int(np.log2(chirp_time)) + 1), 4)
    wfg = WaveformGenerator(
        duration=chirp_time,
        sampling_frequency=args.sampling_frequency,
        start_time=params["geocent_time"] + 0.2 - chirp_time,
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomPv2", minimum_frequency=10.0
        ),
    )

    wf_pols = wfg.time_domain_strain(params)
    window = tukey(len(wfg.time_array), alpha=0.2 / chirp_time)
    wf_pols["plus"] = (
        np.roll(wf_pols["plus"], -int(args.sampling_frequency * 0.2)) * window
    )
    wf_pols["cross"] = (
        np.roll(wf_pols["cross"], -int(args.sampling_frequency * 0.2)) * window
    )
    return wf_pols, wfg.time_array


def do_injections(ifos, injections, channels, strain, args):
    strain_with_injs = strain.copy()
    start_time = ifos.start_time
    end_time = ifos.start_time + ifos.duration
    injections = injections[
        (injections.geocent_time >= start_time) & (injections.geocent_time <= end_time)
    ]
    logger.info(
        f"Adding {len(injections)} injections between {start_time} and {end_time}"
    )
    logger.setLevel("WARNING")
    for inj_no in trange(len(injections)):
        parameters = dict(injections.iloc[inj_no])
        logger.debug(parameters)
        wf_pols, times = td_waveform(parameters, args)

        for ifo in ifos:
            ifo_strain = strain_with_injs[ifo.name]
            channel_name = channels[ifo.name]
            channel = Channel(channel_name)
            time_delay = ifo.time_delay_from_geocenter(
                ra=parameters["ra"],
                dec=parameters["dec"],
                time=parameters["geocent_time"],
            )
            times += time_delay

            signal = np.zeros_like(times)
            for mode in ["plus", "cross"]:
                signal += (
                    ifo.antenna_response(
                        ra=parameters["ra"],
                        dec=parameters["dec"],
                        time=parameters["geocent_time"],
                        psi=parameters["psi"],
                        mode=mode,
                    )
                    * wf_pols[mode]
                )

            inj = ts.TimeSeries(signal, times=times, channel=channel, name=channel_name)
            closest_idx = np.argmin(abs(ifo_strain.xindex.value - inj.xindex.value[0]))
            delta_time = ifo_strain.xindex.value[closest_idx] - inj.xindex.value[0]
            times += delta_time
            inj = ts.TimeSeries(signal, times=times, channel=channel, name=channel_name)
            ifo_strain = ifo_strain.inject(inj)
            times -= time_delay + delta_time
            strain_with_injs[ifo.name] = ifo_strain
    logger.setLevel("INFO")
    return strain_with_injs


def main(args, injections):
    check_directory_exists_and_if_not_mkdir(args.outdir)

    if args.psd_dict == "default":
        psd_dict = dict()
    else:
        psd_dict = convert_string_to_dict(args.psd_dict)

    start_time = args.start_time
    duration = args.frame_duration
    base_channel_name = args.channel_name

    ifos = InterferometerList(args.interferometers)
    for ifo in ifos:
        ifo.minimum_frequency = 10
        if ifo.name in psd_dict:
            if "asd" in psd_dict[ifo.name]:
                method = PowerSpectralDensity.from_amplitude_spectral_density_file
            elif "psd" in psd_dict[ifo.name]:
                method = PowerSpectralDensity.from_power_spectral_density_file
            else:
                logger.warning(
                    f"PSD file name {psd_dict[ifo.name]} not understood, "
                    "assuming it is a PSD not an ASD."
                )
                method = PowerSpectralDensity.from_power_spectral_density_file
            ifo.power_spectral_density = method(psd_dict[ifo.name])
    channels = {ifo.name: ":".join([ifo.name, base_channel_name]) for ifo in ifos}

    strain = dict()

    while start_time < end_time:
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=args.sampling_frequency,
            duration=duration,
            start_time=start_time,
        )
        frame_end_time = start_time + duration
        for ifo in ifos:
            channel_name = channels[ifo.name]
            channel = Channel(channel_name)
            ht = ifo.time_domain_strain
            times = ifo.time_array
            temp = ts.TimeSeries(ht, times=times, channel=channel, name=channel_name)
            try:
                strain[ifo.name] = strain[ifo.name].append(temp, inplace=False)
            except (NameError, ValueError, KeyError):
                strain[ifo.name] = temp

        strain_with_injs = do_injections(
            ifos=ifos,
            injections=injections,
            strain=strain,
            channels=channels,
            args=args,
        )

        for ifo in ifos:
            base_name = f"{args.outdir}/{ifo.name}_BBH_inj_{int(start_time)}_{duration}"
            file_name = f"{base_name}.{args.format}"
            logger.info(f"Saving {file_name}")
            strain_with_injs[ifo.name].write(file_name, format=args.format)
        start_time = frame_end_time


main(args, injections)
