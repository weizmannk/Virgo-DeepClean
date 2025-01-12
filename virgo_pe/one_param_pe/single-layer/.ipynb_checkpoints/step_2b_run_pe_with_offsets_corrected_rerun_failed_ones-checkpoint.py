from utils import *


datadir = "/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Friday_Work/data"
dc_noise_only = os.path.join(datadir, "deepclean-NoiseOnly.gwf")
dc_noise_plus_signal = os.path.join(datadir, "deepclean-NoisePlusSignal.gwf")
org_noise_only = os.path.join(datadir, "original-NoiseOnly.h5")
org_noise_plus_signal = os.path.join(datadir, "original-NoisePlusSignal.h5")
signal_only = os.path.join(datadir, "GW_injected_signal.hdf5")
inj_file = os.path.join(datadir, "INJ-1265127585-4096.csv")
inj_file_json = os.path.join(datadir, "INJ-1265127585-4096.json")


channel = "V1:Hrec_hoft_raw_20000Hz"  # Original Channel channel
channel_DC = "V1:Hrec_hoft_raw_20000Hz_DC"  # Cleanning channel
channel_INJ = "V1:DC_INJ"  # Injected GW signal channel

injections = pd.read_csv(inj_file)
offsets = np.genfromtxt("./offsets.txt")

param_list_full = [
    "chirp_mass",
    "mass_ratio",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    "ra",
    "dec",
    "phase",
    "theta_jn",
    "luminosity_distance",
]

param_list = [
    "chirp_mass",
    "mass_ratio",
    "a_1",
    "a_2",
    "psi",
    "ra",
    "dec",
    "phase",
    "theta_jn",
    "luminosity_distance",
]


for inj_idx in trange(65, 128):

    posteriors = {}
    for parameter in param_list:

        results = pe_on_grid(
            inj_idx=inj_idx,
            theta=parameter,
            injections=injections,
            analysis_data_frame=dc_noise_plus_signal,
            analysis_data_channel=channel_DC,
            psd_data_frame=dc_noise_only,
            psd_data_channel=channel_DC,
            psd_file="./gwpy_psd_deepclean.txt",
            det="V1",
            duration=4,
            postmerger_duration=1,
            psd_duration=512,
            sampling_frequency=2048,
            minimum_frequency=20,
            maximum_frequency=800,
            reference_frequency=20,
            waveform_approximant="IMRPhenomPv2",
            grid_size=600,
            geocent_time_correction=offsets[inj_idx],
        )

        posteriors["grid_" + parameter] = results[0]
        posteriors["pdf_" + parameter] = results[1]
        posteriors["inj_" + parameter] = results[2]

    with open(f"./posteriors/deepclean/results_{inj_idx}.pkl", "wb") as file:
        pickle.dump(posteriors, file)
