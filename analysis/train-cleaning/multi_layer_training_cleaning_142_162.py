import os

# # export CUDA_DEVICE_ORDER=PCI_BUS_ID
# # export CUDA_VISIBLE_DEVICES=1

# # CUDA device order and visible devices
# #for PCdev12
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import deepclean_prod as dc
import pickle
from gwpy.timeseries import TimeSeries, TimeSeriesDict
import glob
import sys
import subprocess
import copy


def get_local_data_and_replace_target(
    cache_file,
    channels,
    t0,
    duration,
    new_target_read_from,
    new_target_channel="V1:Hrec_hoft_raw_20000Hz_DC",
    fs=4096.0,
    chunk_duration=32,
):
    """
    Fetches local data for the specified channels and replaces the target channel with new data.

    Parameters:
    - cache_file (str): The file path to the cache file.
    - channels (list): A list of channel names to fetch.
    - t0 (int): The start time for data fetching.
    - duration (int): The duration for which data is to be fetched.
    - new_target_read_from (str): Directory path to read the new target data from.
    - new_target_channel (str): The name of the new target channel. Defaults to 'V1:Hrec_hoft_raw_20000Hz'.
    - fs (float): The sampling frequency. Defaults to 4096.0 Hz.

    Returns:
    - data (TimeSeriesDataset): The dataset with the target channel replaced by the new target data.
    """
    data = dc.timeseries.TimeSeriesDataset()
    data.get_local_data(
        cache_file,
        channels,
        t0=t0,
        duration=duration,
        fs=fs,
        chunk_duration=chunk_duration,
    )
    new_target_files = glob.glob(f"{new_target_read_from}/*.gwf")
    new_data = TimeSeries.read(
        new_target_files, channel=new_target_channel, start=t0, end=t0 + duration
    )
    data.data[data.target_idx] = new_data
    data.channels[data.target_idx] = new_target_channel
    return data


def in_execute_command(command):
    """
    Executes a given shell command and prints the output.

    Parameters:
    - command (str): The command to execute.
    """
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print(output.decode())

    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.output.decode()}")


def execute_command(command):
    """
    Executes a given shell command and prints the output line by line.

    Parameters:
    - command (str): The command to execute.
    """
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    while True:
        output = process.stdout.readline()
        if not output and process.poll() is not None:
            break
        if output:
            print(output.strip().decode())

    rc = process.poll()
    return rc


CACHE_FILE = "/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/V1-O3b_1265100000-1265399999.cache"
CHANS_LIST_layer0 = "./witnesses_142-162_all.ini"
CHANS_LIST_layer1 = "./witnesses_142-162-layer1.ini"
CHANS_LIST_layer2 = "./witnesses_142-162-layer2.ini"
CHANS_LIST_layer3 = "./witnesses_142-162-layer3.ini"
CHANS_LIST_layer4 = "./witnesses_142-162-layer4.ini"

OUT_CHANNEL_FINAL = "V1:Hrec_hoft_raw_20000Hz_DC"

DEVICE = "cuda"
FS = 4096
TRAIN_T0 = 1265128185
TRAIN_DURATION = 4096
CLEAN_T0 = 1265128185
CLEAN_DURATION = 4096
TRAIN_FAC = 0.9

CHUNK_DURATION = None  # 410

PREFIX = "Hrec-HOFT"
OUT_DIR = "4096"
TRAIN_CADENCE = "4096"
TRAIN_DIR_0 = f"{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}/layer0"
TRAIN_DIR_1 = f"{OUT_DIR}/layer1/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_2 = f"{OUT_DIR}/layer2/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_3 = f"{OUT_DIR}/layer3/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_4 = f"{OUT_DIR}/layer4/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"

# Create training and output directories if they don't exist
directories_to_create = [
    TRAIN_DIR_0,
    TRAIN_DIR_1,
    TRAIN_DIR_2,
    TRAIN_DIR_3,
    TRAIN_DIR_4,
]

for directory in directories_to_create:
    os.makedirs(directory, exist_ok=True)

with open(CHANS_LIST_layer0, "r") as file:
    channels_layer0 = file.read().splitlines()

with open(CHANS_LIST_layer1, "r") as file:
    channels_layer1 = file.read().splitlines()

with open(CHANS_LIST_layer2, "r") as file:
    channels_layer2 = file.read().splitlines()

with open(CHANS_LIST_layer3, "r") as file:
    channels_layer3 = file.read().splitlines()

with open(CHANS_LIST_layer4, "r") as file:
    channels_layer4 = file.read().splitlines()


#  dc-prod-train  --load-dataset True --ifo V1 --chunk-duration None  --save-dataset True --fs 4096 --chanslist ./witnesses_142-162_all.ini --train-kernel 8 --train-stride 0.25 --pad-mode median --filt-fl 142 --filt-fh 162 --filt-order 8 --device cuda --train-frac 0.9 --batch-size 32 --max-epochs 30 --num-workers 4 --lr 1e-3 --weight-decay 1e-5 --fftlength 2 --psd-weight 1.0 --mse-weight 0.0 --train-dir Hrec-HOFT-1265128185-4096/layer0 --train-t0 1265128185 --train-duration 4096


## Download dataset

train_length = int(TRAIN_DURATION * TRAIN_FAC)
data_train = dc.timeseries.TimeSeriesDataset()
data_train.get_local_data(
    CACHE_FILE,
    channels=channels_layer0,
    t0=TRAIN_T0,
    duration=train_length,
    fs=FS,
    chunk_duration=CHUNK_DURATION,
)

# new_frame = f"original-{TRAIN_T0}-{TRAIN_DURATION}.h5"
# data.write(fname=os.path.join(TRAIN_DIR_0, new_frame), group=None, write_mode='w')
data_train.write(
    fname=os.path.join(TRAIN_DIR_0, "training.h5"), group=None, write_mode="w"
)


val_start = TRAIN_T0 + train_length
val_length = TRAIN_DURATION - train_length


data_val = dc.timeseries.TimeSeriesDataset()
data_val.get_local_data(
    CACHE_FILE,
    channels=channels_layer0,
    t0=val_start,
    duration=val_length,
    fs=FS,
    chunk_duration=CHUNK_DURATION,
)

data_val.write(
    fname=os.path.join(TRAIN_DIR_0, "validation.h5"), group=None, write_mode="w"
)

# =====================
# Layer 0/run
# =====================


# #Layer 0 Training
# print("Training for 142-162 Hz band layer0...")
# train_command_layer0 = f"dc-prod-train --cache-file {CACHE_FILE}  --load-dataset True --ifo V1 --chunk-duration {CHUNK_DURATION} --save-dataset True --fs {FS} --chanslist {CHANS_LIST_layer0} --train-kernel 8 --train-stride 0.25 --pad-mode median --filt-fl 142 --filt-fh 162 --filt-order 8 --device {DEVICE} --train-frac {TRAIN_FAC} --batch-size 32 --max-epochs 30 --num-workers 4 --lr 1e-3 --weight-decay 1e-5 --fftlength 2 --psd-weight 1.0 --mse-weight 0.0 --train-dir {TRAIN_DIR_0} --train-t0 {TRAIN_T0} --train-duration {TRAIN_DURATION}"
# execute_command(train_command_layer0)

# #Layer 1 Cleaning
# print("Cleaning for 142-162 Hz band layer0...")
# clean_command_layer0 = f"dc-prod-clean --cache-file {CACHE_FILE} --ifo V1 --chunk-duration {CHUNK_DURATION} --load-dataset True  --save-dataset True --fs {FS} --out-dir {TRAIN_DIR_0} --out-channel {channels_layer1[0]} --chanslist {CHANS_LIST_layer0} --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device {DEVICE} --train-dir {TRAIN_DIR_0} --clean-t0 {CLEAN_T0} --clean-duration {CLEAN_DURATION} --out-file {PREFIX}-{CLEAN_T0}-{CLEAN_DURATION}_layer0.gwf"
# execute_command(clean_command_layer0)


# #=====================--chunk-duration {CHUNK_DURATION}
# # Prepare for Layer 1
# #=====================

# # Dummy target will be replaced with the output of the previous layer
# channels_with_dummy_target_layer1 = copy.deepcopy(channels_layer1)
# channels_with_dummy_target_layer1[0] = channels_layer0[0]

# train_length = int(TRAIN_DURATION * TRAIN_FAC)

# # Fetch data and replace the target for the training set
# data_train = get_local_data_and_replace_target(cache_file=CACHE_FILE, channels=channels_with_dummy_target_layer1,
#                                                t0=TRAIN_T0, duration=train_length,
#                                                new_target_read_from=TRAIN_DIR_0,
#                                                new_target_channel=channels_layer1[0],
#                                                fs=FS, chunk_duration=CHUNK_DURATION)
# data_train.write(fname=os.path.join(TRAIN_DIR_1, 'training.h5'), group=None, write_mode='w')

# # Fetch data and replace the target for the validation set
# val_start = TRAIN_T0 + train_length
# val_length = TRAIN_DURATION - train_length
# data_val = get_local_data_and_replace_target(cache_file=CACHE_FILE, channels=channels_with_dummy_target_layer1,
#                                              t0=val_start, duration=val_length,
#                                              new_target_read_from=TRAIN_DIR_0,
#                                              new_target_channel=channels_layer1[0],
#                                              fs=FS, chunk_duration=CHUNK_DURATION)
# data_val.write(fname=os.path.join(TRAIN_DIR_1, 'validation.h5'), group=None, write_mode='w')

# # =====================
# # Layer 1
# # =====================

# #Layer 1 Training
# print("Training for 142-162 Hz band layer1...")
# train_command_layer1 = f"dc-prod-train --cache-file {CACHE_FILE} --ifo V1 --chunk-duration {CHUNK_DURATION} --load-dataset True  --save-dataset True --fs {FS} --chanslist {CHANS_LIST_layer1} --train-kernel 8 --train-stride 0.25 --pad-mode median --filt-fl 142 --filt-fh 162 --filt-order 8 --device {DEVICE} --train-frac {TRAIN_FAC} --batch-size 32 --max-epochs 30 --num-workers 4 --lr 1e-3 --weight-decay 1e-5 --fftlength 2 --psd-weight 1.0 --mse-weight 0.0 --train-dir {TRAIN_DIR_1} --train-t0 {TRAIN_T0} --train-duration {TRAIN_DURATION}"
# execute_command(train_command_layer1)

# # Preparing data for Cleaning 1
# clean_t0 = CLEAN_T0
# data = get_local_data_and_replace_target(cache_file=CACHE_FILE, channels=channels_with_dummy_target_layer1,
#                                          t0=clean_t0, duration=CLEAN_DURATION,
#                                          new_target_read_from=TRAIN_DIR_0,
#                                          new_target_channel=channels_layer1[0],
#                                          fs=FS, chunk_duration=CHUNK_DURATION)
# new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
# data.write(fname=os.path.join(TRAIN_DIR_1, new_frame), group=None, write_mode='w')

# #Layer 1 Cleaning
# print("Cleaning for 142-162 Hz band layer1...")
# clean_command_layer1 = f"dc-prod-clean --cache-file {CACHE_FILE} --ifo V1 --chunk-duration {CHUNK_DURATION} --load-dataset True  --save-dataset True --fs {FS} --out-dir {TRAIN_DIR_1} --out-channel {channels_layer2[0]} --chanslist {CHANS_LIST_layer1} --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device {DEVICE} --train-dir {TRAIN_DIR_1} --clean-t0 {CLEAN_T0} --clean-duration {CLEAN_DURATION} --out-file {PREFIX}-{CLEAN_T0}-{CLEAN_DURATION}_layer1.gwf"
# execute_command(clean_command_layer1)


# #=====================
# # Prepare for Layer 2
# #=====================

# # Dummy target will be replaced with the output of the previous layer
# channels_with_dummy_target_layer2 = copy.deepcopy(channels_layer2)
# channels_with_dummy_target_layer2[0] = channels_layer1[0]

# train_length = int(TRAIN_DURATION * TRAIN_FAC)

# # Fetch data and replace the target for the training set
# data_train = get_local_data_and_replace_target(cache_file=CACHE_FILE, channels=channels_with_dummy_target_layer2,
#                                                t0=TRAIN_T0, duration=train_length,
#                                                new_target_read_from=TRAIN_DIR_1,
#                                                new_target_channel=channels_layer2[0],
#                                                fs=FS, chunk_duration=CHUNK_DURATION)
# data_train.write(fname=os.path.join(TRAIN_DIR_2, 'training.h5'), group=None, write_mode='w')

# # Fetch data and replace the target for the validation set
# val_start = TRAIN_T0 + train_length
# val_length = TRAIN_DURATION - train_length
# data_val = get_local_data_and_replace_target(cache_file=CACHE_FILE, channels=channels_with_dummy_target_layer2,
#                                              t0=val_start, duration=val_length,
#                                              new_target_read_from=TRAIN_DIR_1,
#                                              new_target_channel=channels_layer2[0],
#                                              fs=FS, chunk_duration=CHUNK_DURATION)
# data_val.write(fname=os.path.join(TRAIN_DIR_2, 'validation.h5'), group=None, write_mode='w')

# # Training for Layer 2

# print("Training for 142-162 Hz band for Layer 2...")
# train_command_layer2 = f"dc-prod-train --cache-file {CACHE_FILE} --load-dataset True --ifo V1 --chunk-duration {CHUNK_DURATION} --save-dataset True --fs {FS} --chanslist {CHANS_LIST_layer2} --train-kernel 8 --train-stride 0.25 --pad-mode median --filt-fl 142 --filt-fh 162 --filt-order 8 --device {DEVICE} --train-frac {TRAIN_FAC} --batch-size 32 --max-epochs 30 --num-workers 4 --lr 1e-3 --weight-decay 1e-5 --fftlength 2 --psd-weight 1.0 --mse-weight 0.0 --train-dir {TRAIN_DIR_2} --train-t0 {TRAIN_T0} --train-duration {TRAIN_DURATION}"
# execute_command(train_command_layer2)


# # Preparing data for Cleaning 2
# clean_t0 = CLEAN_T0
# data = get_local_data_and_replace_target(cache_file=CACHE_FILE, channels=channels_with_dummy_target_layer2,
#                                          t0=clean_t0, duration=CLEAN_DURATION,
#                                          new_target_read_from=TRAIN_DIR_1,
#                                          new_target_channel=channels_layer2[0],
#                                          fs=FS, chunk_duration=CHUNK_DURATION)
# new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
# data.write(fname=os.path.join(TRAIN_DIR_2, new_frame), group=None, write_mode='w')

# # Cleaning for Layer 2
# print("Cleaning for 142-162 Hz band for Layer 2...")
# clean_command_layer2 = f"dc-prod-clean --cache-file {CACHE_FILE} --load-dataset True --ifo V1 --chunk-duration {CHUNK_DURATION} --save-dataset True --fs {FS} --out-dir {TRAIN_DIR_2} --out-channel {channels_layer3[0]} --chanslist {CHANS_LIST_layer2} --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device {DEVICE} --train-dir {TRAIN_DIR_2} --clean-t0 {CLEAN_T0} --clean-duration {CLEAN_DURATION} --out-file {PREFIX}-{CLEAN_T0}-{CLEAN_DURATION}_layer2.gwf"
# execute_command(clean_command_layer2)


# #=====================
# # Prepare for Layer 3
# #=====================

# # Dummy target will be replaced with the output of Layer 2
# channels_with_dummy_target_layer3 = copy.deepcopy(channels_layer3)
# channels_with_dummy_target_layer3[0] = channels_layer1[0]


# # Fetch data and replace the target for the training set for Layer 3
# train_length = int(TRAIN_DURATION * TRAIN_FAC)
# data_train_layer3 = get_local_data_and_replace_target(cache_file=CACHE_FILE, channels=channels_with_dummy_target_layer3,
#                                                       t0=TRAIN_T0, duration=train_length,
#                                                       new_target_read_from=TRAIN_DIR_2,
#                                                       new_target_channel=channels_layer3[0],
#                                                       fs=FS, chunk_duration=CHUNK_DURATION)
# data_train_layer3.write(fname=os.path.join(TRAIN_DIR_3, 'training.h5'), group=None, write_mode='w')

# #Fetch data and replace the target for the validation set for Layer 3
# val_start = TRAIN_T0 + train_length
# val_length = TRAIN_DURATION - train_length
# data_val_layer3 = get_local_data_and_replace_target(cache_file=CACHE_FILE, channels=channels_with_dummy_target_layer3,
#                                                     t0=val_start, duration=val_length,
#                                                     new_target_read_from=TRAIN_DIR_2,
#                                                     new_target_channel=channels_layer3[0],
#                                                     fs=FS, chunk_duration=CHUNK_DURATION)
# data_val_layer3.write(fname=os.path.join(TRAIN_DIR_3, 'validation.h5'), group=None, write_mode='w')

# # Training for Layer 3

# print("Training for 142-162 Hz band for Layer 3...")
# train_command_layer3 = f"dc-prod-train --cache-file {CACHE_FILE} --load-dataset True --ifo V1 --chunk-duration {CHUNK_DURATION} --save-dataset True --fs {FS} --chanslist {CHANS_LIST_layer3} --train-kernel 8 --train-stride 0.25 --pad-mode median --filt-fl 142 --filt-fh 162 --filt-order 8 --device {DEVICE} --train-frac {TRAIN_FAC} --batch-size 32 --max-epochs 30 --num-workers 4 --lr 1e-3 --weight-decay 1e-5 --fftlength 2 --psd-weight 1.0 --mse-weight 0.0 --train-dir {TRAIN_DIR_3} --train-t0 {TRAIN_T0} --train-duration {TRAIN_DURATION}"
# execute_command(train_command_layer3)

# # Preparing data for Cleaning  for layer3
# clean_t0 = CLEAN_T0
# data = get_local_data_and_replace_target(cache_file=CACHE_FILE, channels=channels_with_dummy_target_layer3,
#                                          t0=clean_t0, duration=CLEAN_DURATION,
#                                          new_target_read_from=TRAIN_DIR_2,
#                                          new_target_channel=channels_layer3[0],
#                                          fs=FS, chunk_duration=CHUNK_DURATION)
# new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
# data.write(fname=os.path.join(TRAIN_DIR_3, new_frame), group=None, write_mode='w')

# # Cleaning for Layer 3

# print("Cleaning for 142-162 Hz band for Layer 3...")
# clean_command_layer3 = f"dc-prod-clean --cache-file {CACHE_FILE} --load-dataset True --ifo V1 --chunk-duration {CHUNK_DURATION} --save-dataset True --fs {FS} --out-dir {TRAIN_DIR_3} --out-channel {channels_layer4[0]} --chanslist {CHANS_LIST_layer3} --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device {DEVICE} --train-dir {TRAIN_DIR_3} --clean-t0 {CLEAN_T0} --clean-duration {CLEAN_DURATION} --out-file {PREFIX}-{CLEAN_T0}-{CLEAN_DURATION}_layer3.gwf"
# execute_command(clean_command_layer3)


# #=====================
# # Prepare for Layer 4
# #=====================

# # Dummy target will be replaced with the output of Layer 3
# channels_with_dummy_target_layer4 = copy.deepcopy(channels_layer4)
# channels_with_dummy_target_layer4[0] = channels_layer1[0]

# # Fetch data and replace the target for the training set for Layer 4
# train_length = int(TRAIN_DURATION * TRAIN_FAC)
# data_train_layer4 = get_local_data_and_replace_target(cache_file=CACHE_FILE, channels=channels_with_dummy_target_layer4,
#                                                       t0=TRAIN_T0, duration=train_length,
#                                                       new_target_read_from=TRAIN_DIR_3,
#                                                       new_target_channel=channels_layer4[0],
#                                                       fs=FS, chunk_duration=CHUNK_DURATION)
# data_train_layer4.write(fname=os.path.join(TRAIN_DIR_4, 'training.h5'), group=None, write_mode='w')

# # Fetch data and replace the target for the validation set for Layer 4
# val_start = TRAIN_T0 + train_length
# val_length = TRAIN_DURATION - train_length
# data_val_layer4 = get_local_data_and_replace_target(cache_file=CACHE_FILE, channels=channels_with_dummy_target_layer4,
#                                                     t0=val_start, duration=val_length,
#                                                     new_target_read_from=TRAIN_DIR_3,
#                                                     new_target_channel=channels_layer4[0],
#                                                     fs=FS)
# data_val_layer4.write(fname=os.path.join(TRAIN_DIR_4, 'validation.h5'), group=None, write_mode='w')


# # Training for Layer 4

# print("Training for 142-162 Hz band for Layer 4...")
# train_command_layer4 = f"dc-prod-train --cache-file {CACHE_FILE} --load-dataset True --ifo V1 --chunk-duration {CHUNK_DURATION} --save-dataset True --fs {FS} --chanslist {CHANS_LIST_layer4} --train-kernel 8 --train-stride 0.25 --pad-mode median --filt-fl 142 --filt-fh 162 --filt-order 8 --device {DEVICE} --train-frac {TRAIN_FAC} --batch-size 32 --max-epochs 30 --num-workers 4 --lr 1e-3 --weight-decay 1e-5 --fftlength 2 --psd-weight 1.0 --mse-weight 0.0 --train-dir {TRAIN_DIR_4} --train-t0 {TRAIN_T0} --train-duration {TRAIN_DURATION}"
# execute_command(train_command_layer4)

# clean_t0 = CLEAN_T0
# data = get_local_data_and_replace_target(cache_file=CACHE_FILE, channels=channels_with_dummy_target_layer4,
#                                          t0=clean_t0, duration=CLEAN_DURATION,
#                                          new_target_read_from=TRAIN_DIR_3,
#                                          new_target_channel=channels_layer4[0],
#                                          fs=FS, chunk_duration=CHUNK_DURATION)
# new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
# data.write(fname=os.path.join(TRAIN_DIR_4, new_frame), group=None, write_mode='w')

# # Cleaning for Layer 4

# print("Cleaning for 142-162 Hz band for Layer 4...")
# clean_command_layer4 = f"dc-prod-clean --cache-file {CACHE_FILE} --load-dataset True --ifo V1 --chunk-duration {CHUNK_DURATION} --save-dataset True --fs {FS} --out-dir {TRAIN_DIR_4} --out-channel {OUT_CHANNEL_FINAL} --chanslist {CHANS_LIST_layer4} --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device {DEVICE} --train-dir {TRAIN_DIR_4} --clean-t0 {CLEAN_T0} --clean-duration {CLEAN_DURATION} --out-file {PREFIX}-{CLEAN_T0}-{CLEAN_DURATION}_layer4.gwf"
# execute_command(clean_command_layer4)
