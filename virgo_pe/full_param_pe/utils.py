import os
import numpy as np
import pandas as pd
import sys
import glob
import pickle
import scipy.stats as ss
import time
import copy
import bilby
from gwpy.timeseries import TimeSeries

import lalsimulation as lalsim
import lal
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.prior import BBHPriorDict
from bilby.gw.source import lal_binary_black_hole
from tqdm.auto import trange


def get_gwpy_mean_psd (psd_data, fs = 4096, 
                      fftlength=32,
                      low_frequency = 16,
                      method = 'welch'):
    psd = psd_data.psd(fftlength=fftlength,overlap=0.0,method=method)
    return np.array(psd.frequencies), np.array(psd.value)


def pe_on_grid (inj_idx,
                 theta, 
                 injections,
                 analysis_data_frame,
                 analysis_data_channel,
                 psd_data_frame,
                 psd_data_channel,
                 psd_file,
                 det = "V1",
                 duration = 4,
                 postmerger_duration = 1,
                 psd_duration = 512,
                 sampling_frequency = 2048,
                 minimum_frequency = 20,
                 maximum_frequency = 800,
                 reference_frequency = 20,
                 waveform_approximant="IMRPhenomPv2",
                 grid_size = 300,
                 geocent_time_correction = 0,
                 phase_correction = 0,
                 ra_correction = 0
        ):

    injection_parameters = injections.iloc[inj_idx].to_dict()
    injection_parameters['geocent_time'] =  injection_parameters['geocent_time'] + geocent_time_correction
    injection_parameters['phase'] =  injection_parameters['phase'] + phase_correction
    injection_parameters['ra'] =  injection_parameters['ra'] + ra_correction
    param_dict = injection_parameters

    mc_inj = bilby.gw.conversion.component_masses_to_chirp_mass(injection_parameters['mass_1'], injection_parameters['mass_2'])
    z = bilby.gw.conversion.luminosity_distance_to_redshift(injection_parameters['luminosity_distance'])
    mass_ratio_inj = bilby.gw.conversion.component_masses_to_mass_ratio(injection_parameters['mass_1'], injection_parameters['mass_2'])

    trigger_time= injections['geocent_time'][inj_idx]
    premerger_duration = duration - postmerger_duration
    start_time = trigger_time - premerger_duration
    end_time   = start_time + duration
    psd_start_time = start_time + 8
    psd_end_time = psd_start_time + psd_duration
    #trigger_time = end_time - 1

    if analysis_data_channel == None:
        analysis_data = TimeSeries.read(
            analysis_data_frame, 
            start = start_time, 
            end = end_time
        )
    else:
        analysis_data = TimeSeries.read(
            analysis_data_frame, 
            analysis_data_channel,
            start = start_time, 
            end = end_time
        )

    waveform_arguments=dict(minimum_frequency=10.0)
    params_new, _ = convert_to_lal_binary_black_hole_parameters(injection_parameters)
    
    print(f"tilt_1 : {params_new['a_1'] * np.cos(params_new['tilt_1'])} ,  tilt_2 : {params_new['a_2'] * np.cos(params_new['tilt_2'])}")
                                                                                                              
    chirp_time = lalsim.SimInspiralChirpTimeBound(
        10,
        params_new["mass_1"] * lal.MSUN_SI,
        params_new["mass_2"] * lal.MSUN_SI,
        params_new["a_1"] * np.cos(params_new["tilt_1"]),
        params_new["a_2"] * np.cos(params_new["tilt_2"]),
    )
    chirp_time = max(2 ** (int(np.log2(chirp_time)) + 1), duration)
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=chirp_time,
        sampling_frequency=sampling_frequency,
        start_time=injection_parameters["geocent_time"] + 0.2 - chirp_time,
        # the 0.2 is because it has been used in the original injections
        frequency_domain_source_model=lal_binary_black_hole,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(minimum_frequency=10.0),
    )

    ifos = bilby.gw.detector.InterferometerList([det])
    ifos[0].strain_data.set_from_gwpy_timeseries(analysis_data)
    
    if not os.path.exists(psd_file):
        # first make it exist!
        if psd_data_frame is not None:
            if psd_data_channel == None:
                psd_data = TimeSeries.read(
                    psd_data_frame, 
                    start = psd_start_time, 
                    end = psd_end_time
                )
            else:
                psd_data = TimeSeries.read(
                    psd_data_frame, psd_data_channel, 
                    start = psd_start_time, 
                    end = psd_end_time
                )

            psd_gwpy = get_gwpy_mean_psd (
                psd_data, 
                fs = sampling_frequency, 
                low_frequency=minimum_frequency,
                fftlength=32, 
                method = 'welch'
            )

            np.savetxt(
                fname = psd_file, 
                X = np.array([ psd_gwpy[0][:-1], psd_gwpy[1][:-1] ]).T
            )
    
    
    ifos[0].power_spectral_density.psd_file = psd_file

    priors = BBHPriorDict()
    # First, set all the priors to the injected values, no widths at all for any parameter.
    priors['chirp_mass']  = mc_inj
    priors['mass_ratio']  = mass_ratio_inj
    for key in ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'psi', 'ra',
                'dec', 'phase', 'theta_jn', 'luminosity_distance', 'geocent_time']:
        priors[key] = injection_parameters[key]

    # Next, rewrite the prior for the search parameter with a prior width.  
    if theta == "chirp_mass":
        theta_min = max(5, mc_inj - 0.05)
        theta_max = min(90, mc_inj + 0.05)

        priors[theta]  = bilby.core.prior.Uniform(name=theta, minimum=theta_min, maximum=theta_max, unit='$M_{\odot}$', boundary='reflective')

    if theta == "mass_ratio":
        theta_min = mass_ratio_inj - 0.05
        theta_max = mass_ratio_inj + 0.05
        
        priors[theta]  = bilby.core.prior.Uniform(name=theta, 
                            minimum=max(0.2, theta_min), 
                            maximum=min(1, theta_max), 
                            boundary='reflective')
    
    if (theta == "a_1") | (theta == "a_2"):
        theta_min = injection_parameters[theta] - 0.03
        theta_max = injection_parameters[theta] + 0.03

        priors[theta]  = bilby.core.prior.Uniform(name=theta, 
                            minimum=max(0.0, theta_min), 
                            maximum=min(0.99, theta_max), 
                            boundary='reflective')

    if (theta == "phi_12") | (theta == "phi_jl")| (theta == "psi")| (theta == "phase")| (theta == "ra"):
        theta_min = injection_parameters[theta] - 0.1
        theta_max = injection_parameters[theta] + 0.1

        priors[theta]  = bilby.core.prior.Uniform(name=theta, 
                            minimum=max(0.0, theta_min), 
                            maximum=min(2*np.pi, theta_max), 
                            boundary='periodic')

    if (theta == "theta_jn")| (theta == "tilt_1")| (theta == "tilt_2"):
        theta_min = injection_parameters[theta] - 0.1
        theta_max = injection_parameters[theta] + 0.1

        priors[theta]  = bilby.core.prior.Uniform(name=theta, 
                            minimum=max(0.0, theta_min), 
                            maximum=min(np.pi, theta_max), 
                            boundary='periodic')

    if (theta == "dec"):
        theta_min = injection_parameters[theta] - 0.1
        theta_max = injection_parameters[theta] + 0.1

        priors[theta]  = bilby.core.prior.Uniform(name=theta, 
                            minimum=max(-np.pi/2, theta_min), 
                            maximum=min(np.pi/2, theta_max), 
                            boundary='periodic')

        
    if (theta == "luminosity_distance"):
        theta_min = injection_parameters[theta] - 5
        theta_max = injection_parameters[theta] + 5

        priors[theta]  = bilby.core.prior.Uniform(name=theta, 
                            minimum=max(2, theta_min), 
                            maximum=min(40, theta_max), 
                            boundary='reflective')

    if (theta == "geocent_time"):
        theta_min = injection_parameters[theta] - 0.001
        theta_max = injection_parameters[theta] + 0.001

        priors[theta]  = bilby.core.prior.Uniform(name=theta, 
                            minimum=theta_min, 
                            maximum=theta_max, 
                            boundary='reflective')
        
    likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos, 
            waveform_generator=waveform_generator,
            reference_frame= 'sky',
            time_reference= 'geocent',
            time_marginalization = False,
            distance_marginalization = False, 
            jitter_time= False,
            phase_marginalization = False,
            priors = priors)    
    
    grid = np.linspace(priors[theta].minimum, priors[theta].maximum,grid_size)
    LLR  = np.zeros(grid.shape)

    if theta == 'chirp_mass':
        theta_inj = mc_inj
        for i in trange(len(LLR)):
            mtot = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(
                         grid[i],mass_ratio_inj)
            m1, m2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
                         mass_ratio_inj, mtot)
            param_dict['mass_1'] = m1
            param_dict['mass_2'] = m2
            likelihood.parameters = param_dict

            LLR[i] = likelihood.log_likelihood_ratio()

    if theta == 'mass_ratio':
        theta_inj = mass_ratio_inj
        for i in trange(len(LLR)):
            mtot = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(
                      mc_inj, grid[i])
            m1, m2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(
                      grid[i], mtot)
            param_dict['mass_1'] = m1
            param_dict['mass_2'] = m2
            likelihood.parameters = param_dict

            LLR[i] = likelihood.log_likelihood_ratio()

    if (theta != "chirp_mass") and (theta != "mass_ratio"):
        theta_inj = injection_parameters[theta]
        for i in trange(len(LLR)):
            param_dict[theta] = grid[i]
            likelihood.parameters = param_dict

            LLR[i] = likelihood.log_likelihood_ratio()


    return grid, np.exp(LLR-max(LLR)), theta_inj




