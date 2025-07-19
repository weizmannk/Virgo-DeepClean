#!/cvmfs/software.igwn.org/conda/envs/igwn/bin/python

import os
import numpy as np
import pandas as pd
import sys
import glob
import h5py
import pickle
import scipy
import time
import copy
import bilby
from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey
from itertools import product
import lalsimulation as lalsim
import lal
#import tbs

from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.prior import BBHPriorDict
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.detector import PowerSpectralDensity
from tqdm.auto import trange
from scipy.special import logsumexp

def timeSeries_from_h5 (filename, 
                        channel = 'H1:GDS-CALIB_STRAIN' ):

    """
    To read the .h5 output files produced by dc-prod-clean
    this format is used for storing the original (unclean) data
    NB: Cleaned data are stored as frame files

    Parameters
    ----------
    filename : `str`
        path to the h5 file to be read
    channel  : `str`
        default is 'H1:GDS-CALIB_STRAIN'
    
    Returns:
    --------
    data_ts : `gwpy.timeseries.TimeSeries`
        the data in the timeseries format.

    """

    f = h5py.File(filename, 'r')
    channels = list(f.keys())

    self_channels = []
    self_data = []
    with h5py.File(filename, 'r') as f:
        fobj = f
        for chan, data in fobj.items():
            if chan not in channels:
                continue
            self_channels.append(chan)
            self_data.append(data[:])
            t0 = data.attrs['t0']
            fs = data.attrs['sample_rate']

    data_ndarray = self_data[self_channels.index(channel)]
    data_ts = TimeSeries(data_ndarray, t0=t0, sample_rate=fs, name=channel, unit="ct", channel=channel)
    
    return data_ts


# we will have only 128 pids for one frame file of 4096 seconds and injections spaced by 32 seconds
pid = int(sys.argv[1])
outdir = "outdir_org"
label=f"virgo_deepclean_pe_injection_{str(pid)}"

# sampler settings
sampler_kwargs={'nlive': 1000, 'walks': 100, 'check_point_plot': True, 
                'check_point_delta_t': 1800, 'print_method': 'interval-60', 
                'nact': 3, 'first_update': {'min_ncall': 300000}}

# all the paths required re here  =========

datadir = "/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Friday_Work/15_415_Hz/data"
inj_file             = os.path.join(datadir, 'INJ-1265127585-4096.csv')
channel_org = 'V1:Hrec_hoft_raw_20000Hz'  # Original Channel channel
datafile_org_NoiseOnly       = os.path.join(datadir, 'original-NoiseOnly.h5')
datafile_org_NoisePlusSignal = os.path.join(datadir, 'original-NoisePlusSignal.h5')

#datafile_dc_NoiseOnly        = os.path.join(datadir, 'deepclean-NoiseOnly.gwf')
#datafile_dc_NoisePlusSignal  = os.path.join(datadir, 'deepclean-NoisePlusSignal.gwf')
#signal_only          = os.path.join(datadir, 'GW_injected_signal.hdf5')
#inj_file_json        = os.path.join(datadir, 'INJ-1265127585-4096.json')
#channel_dc="V1:Hrec_hoft_raw_20000Hz_DC"  # Cleanning channel
#channel_INJ = "V1:DC_INJ"                # Injected GW signal channel
## 



## data selection based on the pid
frame_start_time = 1265127585
frame_end_time = 1265127585 + 4096

injections = pd.read_csv(inj_file)
injection_times = injections['geocent_time'].to_list()

trigtime = injection_times[pid]


# loaded_data = TimeSeries.read(
#     datafile_org_NoisePlusSignal, 
#     channel = channel_org, 
#     start = frame_start_time, end = frame_end_time)

loaded_data = timeSeries_from_h5(datafile_org_NoisePlusSignal, channel_org)

##########################################################################
##########################################################################
##########################################################################

#frame_duration = 4096  
duration = 8
post_trigger_duration = 2
end_time = trigtime +  post_trigger_duration
start_time = end_time - duration

sampling_frequency = 4096
minimum_frequency = 15
maximum_frequency = 1024
reference_frequency = 20
prior_file = "./prior.prior"
inject = False


waveform_approximant="IMRPhenomPv2"
psd_filename = "data/psd_org.txt"
det = "V1"
psd = np.loadtxt(psd_filename)[:,1]

#########################################################

priors = bilby.gw.prior.BBHPriorDict(prior_file)


deltaT = 0.3

# priors['geocent_time'] = bilby.gw.prior.DeltaFunction(
#     peak=trigtime, name=None, latex_label=None, unit=None)
priors['geocent_time'] = bilby.gw.prior.Uniform(minimum=trigtime-deltaT, 
        maximum=trigtime+deltaT, name='geocent_time')



waveform_arguments=dict(minimum_frequency=minimum_frequency, 
                        maximum_frequency = maximum_frequency)
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=start_time,
    frequency_domain_source_model=lal_binary_black_hole,
    parameter_conversion=convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments
)

ifos = bilby.gw.detector.InterferometerList([det])
ifos[0].minimum_frequency = minimum_frequency
ifos[0].maximum_frequency = maximum_frequency
ifos[0].power_spectral_density.psd_file = psd_filename
td_data = loaded_data.crop(start_time, end_time)
ifos[0].strain_data.set_from_gwpy_timeseries(td_data)

injection_parameters = None
if inject:
    injection = pd.DataFrame(priors.sample(1))
    injection_parameters = injection.iloc[0].to_dict() # unique injection
    ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters)

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

likelihood_DPM = bilby.gw.GravitationalWaveTransient(
                interferometers=ifos, 
                waveform_generator=waveform_generator,
                reference_frame= 'sky',
                time_reference= 'geocent',
                time_marginalization = True,
                distance_marginalization = True, 
                jitter_time= True,
                phase_marginalization = True,
                priors = priors)



result = bilby.run_sampler(
    likelihood=likelihood_DPM, priors=priors, sampler='dynesty',
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    injection_parameters=injection_parameters, outdir=outdir,
    label=label,dlogz=0.1, 
    **sampler_kwargs)




