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

OUT_CHANNEL_FINAL = "V1:Hrec_hoft_raw_20000Hz_DC"

DEVICE = "cuda"
FS = 4096
TRAIN_T0 = 1265276737
TRAIN_DURATION = 4096
CLEAN_T0 = 1265276737
CLEAN_DURATION = 4096
TRAIN_FAC = 0.9

CHUNK_DURATION = None  # 410

PREFIX = "Hrec-HOFT"
OUT_DIR = "4096"
TRAIN_CADENCE = "50000"


TRAIN_DIR_0 = f"{OUT_DIR}/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"


TRAIN_DIR_1 = f"{OUT_DIR}/layer1/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_2 = f"{OUT_DIR}/layer2/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_3 = f"{OUT_DIR}/layer3/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_4 = f"{OUT_DIR}/layer4/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"

# Create training and output directories if they don't exist
directories_to_create = [TRAIN_DIR_0]

for directory in directories_to_create:
    os.makedirs(directory, exist_ok=True)

with open(CHANS_LIST_layer0, "r") as file:
    channels_layer0 = file.read().splitlines()

# with open(CHANS_LIST_layer1, 'r') as file:
#     channels_layer1 = file.read().splitlines()

# with open(CHANS_LIST_layer2, 'r') as file:
#     channels_layer2 = file.read().splitlines()

# with open(CHANS_LIST_layer3, 'r') as file:
#     channels_layer3 = file.read().splitlines()

# with open(CHANS_LIST_layer4, 'r') as file:
#     channels_layer4 = file.read().splitlines()


#  dc-prod-train  --load-dataset True --ifo V1 --chunk-duration None  --save-dataset True --fs 4096 --chanslist ./witnesses_142-162_all.ini --train-kernel 8 --train-stride 0.25 --pad-mode median --filt-fl 142 --filt-fh 162 --filt-order 8 --device cuda --train-frac 0.9 --batch-size 32 --max-epochs 30 --num-workers 4 --lr 1e-3 --weight-decay 1e-5 --fftlength 2 --psd-weight 1.0 --mse-weight 0.0 --train-dir Hrec-HOFT-1265128185-4096/layer0 --train-t0 1265128185 --train-duration 4096


## Download dataset

# s1=1265127585; t1=1265225889
# s2=1265227585; t2=1265325889
# s3=1265327585; t3=1265352161


# t1=1265352161

# train_t0 = 1265327585

# while train_t0<=t1:
#     train_length =  TRAIN_DURATION
#     data_train = dc.timeseries.TimeSeriesDataset()
#     data_train.get_local_data(CACHE_FILE, channels=channels_layer0, t0=train_t0, duration=train_length, fs=FS, chunk_duration=CHUNK_DURATION)

#     data_train.write(fname=os.path.join(TRAIN_DIR_0, f'original-{train_t0}-4096.h5'), group=None, write_mode='w')

#     train_t0 += train_length
#     print(train_t0)


TRAIN_T0 = 1265276737  # train_t0


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
