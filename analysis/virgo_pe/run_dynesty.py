#!/home/muhammed.saleem/.conda/envs/tbs-igwn-py39/bin/python

import os
import numpy as np
import pandas as pd
import sys
import glob
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
import tbs

from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.prior import BBHPriorDict
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.detector import PowerSpectralDensity
from tqdm.auto import trange
from scipy.special import logsumexp

# we will have only 128 pids for one frame file of 4096 seconds and injections spaced by 32 seconds
pid = int(sys.argv[1])
outdir = "outdir_org"
label = f"virgo_deepclean_pe_injection_{str(pid)}"

# sampler settings
sampler_kwargs = {
    "nlive": 1000,
    "walks": 100,
    "check_point_plot": True,
    "check_point_delta_t": 1800,
    "print_method": "interval-60",
    "nact": 3,
    "first_update": {"min_ncall": 300000},
}


## data selection based on the pid
injections_csv = (
    "/home/weizmann.kiendrebeogo/DeepClean/saleem-Dc/INJ-1265127585-4096.csv"
)
start_time = 1265127585
end_time = 1265127585 + 4096

injections = pd.read_csv(injections_csv)
injection_times = injections["geocent_time"].to_list()
trigtime = injection_times[pid]


datadir = "/home/muhammed.saleem/deepClean/deepclean/virgo_pe/data"

datafile_dc_NoiseOnly = os.path.join(datadir, "deepclean-NoiseOnly.gwf")
datafile_dc_NoisePlusSignal = os.path.join(datadir, "deepclean-NoisePlusSignal.gwf")

datafile_org_NoiseOnly = os.path.join(datadir, "original-NoiseOnly.gwf")
datafile_org_NoisePlusSignal = os.path.join(datadir, "original-NoisePlusSignal.gwf")

channel_org = "V1:Hrec_hoft_raw_20000Hz"
channel_dc = "V1:Hrec_hoft_raw_20000Hz_DC"

loaded_data = TimeSeries.read(
    datafile_dc_NoisePlusSignal, channel=channel_dc, start=start_time, end=end_time
)


##########################################################################
##########################################################################
##########################################################################

# frame_duration = 4096
duration = 8
post_trigger_duration = 1
sampling_frequency = 4096
minimum_frequency = 20
maximum_frequency = 800
reference_frequency = 20
prior_file = "./prior.prior"
inject = False


waveform_approximant = "IMRPhenomPv2"
psd_filename = "data/psd_dc.txt"
det = "V1"
masked_finite_psd = np.loadtxt(psd_filename)[:, 1]

#########################################################

priors = bilby.gw.prior.BBHPriorDict(prior_file)


## we are here and need to clarify how the times are defined start means start of what?

trigtime = end_time - post_trigger_duration
psd_start_time = end_time
psd_end_time = psd_start_time + psd_duration


deltaT = 0.5

# priors['geocent_time'] = bilby.gw.prior.DeltaFunction(
#     peak=trigtime, name=None, latex_label=None, unit=None)
priors["geocent_time"] = bilby.gw.prior.Uniform(
    minimum=trigtime - deltaT, maximum=trigtime + deltaT, name="geocent_time"
)


waveform_arguments = dict(
    minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency
)
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=start_time,
    frequency_domain_source_model=lal_binary_black_hole,
    parameter_conversion=convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

ifos = bilby.gw.detector.InterferometerList([det])
ifos[0].minimum_frequency = minimum_frequency
ifos[0].maximum_frequency = maximum_frequency
ifos[
    0
].power_spectral_density.psd_file = (
    "/home/muhammed.saleem/tbs/codes/td_analysis/exp21/GW170817_non_windowed_psd.txt"
)
td_data = loaded_data.crop(start_time, end_time)
ifos[0].strain_data.set_from_gwpy_timeseries(td_data)

injection_parameters = None
if inject:
    injection = pd.DataFrame(priors.sample(1))
    injection_parameters = injection.iloc[0].to_dict()  # unique injection
    ifos.inject_signal(
        waveform_generator=waveform_generator, parameters=injection_parameters
    )

likelihood_diag = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    reference_frame="sky",
    time_reference="geocent",
    time_marginalization=False,
    distance_marginalization=False,
    jitter_time=False,
    phase_marginalization=False,
    priors=priors,
)

likelihood_diag_DPM = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    reference_frame="sky",
    time_reference="geocent",
    time_marginalization=False,
    distance_marginalization=True,
    jitter_time=False,
    phase_marginalization=True,
    priors=priors,
)


likelihood_nondiag = tbs.covariance_likelihood.FiniteDurationGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    inverse_covariance_matrix_dict=inverse_covariance_matrix_dict,
    reference_frame="sky",
    time_reference="geocent",
    time_marginalization=False,
    distance_marginalization=False,
    jitter_time=False,
    phase_marginalization=False,
    priors=priors,
)

outfile_reweighted = f"{outdir}/{label}_reweighted.p"

if os.path.exists(outfile_reweighted):
    print("Re-weighted file exists . .  Quiting the job . . ")
    quit()
else:
    print("Re-weighted file not found . .  Computing . . ")


result = bilby.run_sampler(
    likelihood=likelihood_diag_DPM,
    priors=priors,
    sampler="dynesty",
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    dlogz=0.1,
    **sampler_kwargs,
)


log_rwt, log_wi = tbs.pp_utils.reweight(result, likelihood_nondiag, likelihood_diag)
logb_nondiag = result.log_bayes_factor + log_rwt
dict_nondiag = {}
dict_nondiag["log_weights"] = log_wi
dict_nondiag["log_bayes_factor_nondiag"] = logb_nondiag
dict_nondiag["log_bayes_factor_diag"] = result.log_bayes_factor
dict_nondiag["log_rwt"] = log_rwt

pickle.dump(dict_nondiag, open(outfile_reweighted, "wb"))
