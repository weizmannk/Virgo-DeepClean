# python add_injection_strain_to_noise_strain.py --strain-channel V1:Hrec_hoft_raw_20000Hz --chanslist /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/98-110_Hz/98-110_best_one.ini  --noise-frame /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/98-110_Hz/4096/Hrec-HOFT-1265127585-100000/original-1265127585-4096.h5 --inj-frame /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Signal_injection/injection_Dir_98_108_Hz/V1_BBH_inj_1265127585.0_1265128609.0.hdf5 --output-frame /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Signal_injection/injection_Dir_98_108_Hz

#!/usr/bin/env python

import deepclean_prod as dc
import numpy as np
from gwpy.timeseries import TimeSeries
from argparse import ArgumentParser
import os

parser = ArgumentParser("Add the injection strain onto the noise strain.")
parser.add_argument("--noise-frame", type=str, help="Path to the noise frame file")
parser.add_argument("--inj-frame", type=str, help="Path to the injection strain frame file")
parser.add_argument("--output-frame", type=str, help="Path to the noise+signal frame file")
parser.add_argument("--chanslist", type=str, help="Path to the chanslist that lists the channels in --noise-frame")
parser.add_argument("--strain-channel", type=str, help="Strain channel name")

args = parser.parse_args()


# Read the data and the injection strain
data = dc.timeseries.TimeSeriesSegmentDataset(8,1)
data.read(args.noise_frame, args.chanslist)
target_id = np.where(data.channels == args.strain_channel)[0][0]

# Add the injection strain to the noise strain
inj_ts   = TimeSeries.read(args.inj_frame)
data.data[target_id] = data.data[target_id] + inj_ts.value

# Assuming args.output_frame is a directory path and not a file path
output_dir = args.output_frame
print(f"Attempting to create output directory: {output_dir}")
os.makedirs(output_dir, exist_ok=True)
if os.path.isdir(output_dir):
    print(f"Directory created successfully: {output_dir}")
else:
    print(f"Failed to create directory: {output_dir}")

injected_file = os.path.join(args.output_frame,  f"injected_{(args.noise_frame).split('/')[-1]}")
print(injected_file)
# Write the combined data to the output frame
data.write(injected_file)