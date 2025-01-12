#!/usr/bin/env python


# dc-prod-clean  --cache-file /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/V1-O3b_1265100000-1265399999.cache --load-dataset False --ifo V1 --save-dataset True --chunk-duration None --fs 4096 --out-dir 4096/Hrec-HOFT-1265127585-100000 --out-channel V1:Hrec_hoft_raw_20000Hz_DC --chanslist ./witnesses_295-305_Hz.ini  --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device cuda --train-dir Hrec-HOFT-1265128185-100000  --clean-t0 1265127585 --clean-duration 1024 --out-file Hrec-HOFT-1265127585-4096.gwf


# dc-prod-train --cache-file /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/V1-O3b_1265100000-1265399999.cache  --load-dataset False --ifo V1 --chunk-duration None  --save-dataset True --fs 4096 --chanslist ./witnesses_295-305_Hz.ini --train-kernel 8 --train-stride 0.25 --pad-mode median --filt-fl 295 --filt-fh 305 --filt-order 8 --device cuda --train-frac 0.9 --batch-size 32 --max-epochs 30 --num-workers 4 --lr 1e-3 --weight-decay 1e-5 --fftlength 2 --psd-weight 1.0 --mse-weight 0.0 --train-dir  Hrec-HOFT-1265128185-100000 --train-t0 1265127585 --train-duration 1024

"""
---------------------------------------------------------------------------------------------------
ABOUT THE SCRIPT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository URL  : https://github.com/weizmannk/Virgo-DeepClean.git
Creation Date   : Febuary 2024
Description     :


"""

import os

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=1

# # CUDA device order and visible devices
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
    data.get_local_data(cache_file, channels, t0=t0, duration=duration, fs=fs)
    new_target_files = glob.glob(f"{new_target_read_from}/*{t0}*.gwf")
    new_data = TimeSeries.read(
        new_target_files, channel=new_target_channel, start=t0, end=t0 + duration
    )
    data.data[data.target_idx] = new_data
    data.channels[data.target_idx] = new_target_channel
    return data


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


def train_layer(
    cache_file,
    chans_list_layer,
    flow,
    fhigh,
    train_t0,
    train_duration,
    train_dir,
    load_dataset=False,
    train_fac=0.9,
    fs=4096,
    device="cuda",
    ifo="V1",
):
    train_command_layer = f"dc-prod-train  --cache-file {cache_file} --ifo {ifo} --load-dataset {load_dataset} --save-dataset True --fs {fs} --chanslist {chans_list_layer} --train-kernel 8 --train-stride 0.25 --pad-mode median --filt-fl {flow} --filt-fh {fhigh} --filt-order 8 --device {device} --train-frac {train_fac} --batch-size 32 --max-epochs 30 --num-workers 4 --lr 1e-3 --weight-decay 1e-5 --fftlength 2 --psd-weight 1.0 --mse-weight 0.0 --train-dir {train_dir} --train-t0 {train_t0} --train-duration {train_duration}"

    execute_command(train_command_layer)
    # return train_command_layer


def clean_layer(
    cache_file,
    chans_list_layer,
    train_dir,
    clean_t0,
    clean_duration,
    out_channel_layer,
    layer_num,
    load_dataset=False,
    fs=4096,
    device="cuda",
    ifo="V1",
    prefix="Hrec-HOFT",
):
    clean_command_layer = f"dc-prod-clean --cache-file {cache_file} --ifo {ifo} --load-dataset {load_dataset} --save-dataset True --fs {fs} --out-dir {train_dir} --out-channel {out_channel_layer} --chanslist {chans_list_layer} --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device {device} --train-dir {train_dir} --clean-t0 {clean_t0} --clean-duration {clean_duration} --out-file {prefix}-{clean_t0}-{clean_duration}_{layer_num}.gwf"

    execute_command(clean_command_layer)
    # return clean_command_layer


CACHE_FILE = "/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/V1-O3b_1265100000-1265399999.cache"

CHANS_LIST_layer0 = "./witnesses/witnesses_142-162_Hz.ini"
CHANS_LIST_layer1 = "./witnesses/witnesses_15-20_Hz.ini"
CHANS_LIST_layer2 = "./witnesses/witnesses_33-39_Hz.ini"
CHANS_LIST_layer3 = "./witnesses/witnesses_55-65_Hz.ini"
CHANS_LIST_layer4 = "./witnesses/witnesses_75-80_Hz.ini"
CHANS_LIST_layer5 = "./witnesses/witnesses_98-110_Hz.ini"
CHANS_LIST_layer6 = "./witnesses/witnesses_137-139_Hz.ini"
CHANS_LIST_layer7 = "./witnesses/witnesses_197-208_Hz.ini"
CHANS_LIST_layer8 = "./witnesses/witnesses_247-252_Hz.ini"
CHANS_LIST_layer9 = "./witnesses/witnesses_295-305_Hz.ini"
CHANS_LIST_layer10 = "./witnesses/witnesses_345-355_Hz.ini"
CHANS_LIST_layer11 = "./witnesses/witnesses_355-367_Hz.ini"
CHANS_LIST_layer12 = "./witnesses/witnesses_395-415_Hz.ini"
CHANS_LIST_layer13 = "./witnesses/witnesses_548-555_Hz.ini"
CHANS_LIST_layer14 = "./witnesses/witnesses_598-603_Hz.ini"

IFO = "V1"
DEVICE = "cuda"
FS = 4096
TRAIN_T0 = 1265127585
TRAIN_DURATION = 4096

CLEAN_DURATION = 4096
TRAIN_FAC = 0.9

PREFIX = "Hrec-HOFT"
OUT_DIR = "4096"
TRAIN_CADENCE = "100000"

TRAIN_DIR_0 = f"{OUT_DIR}/layer0/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_1 = f"{OUT_DIR}/layer1/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_2 = f"{OUT_DIR}/layer2/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_3 = f"{OUT_DIR}/layer3/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_4 = f"{OUT_DIR}/layer4/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_5 = f"{OUT_DIR}/layer5/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_6 = f"{OUT_DIR}/layer6/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_7 = f"{OUT_DIR}/layer7/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_8 = f"{OUT_DIR}/layer8/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_9 = f"{OUT_DIR}/layer9/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_10 = f"{OUT_DIR}/layer10/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_11 = f"{OUT_DIR}/layer11/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_12 = f"{OUT_DIR}/layer12/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_13 = f"{OUT_DIR}/layer13/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
TRAIN_DIR_14 = f"{OUT_DIR}/layer14/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"

# Create training and output directories if they don't exist
directories_to_create = [
    TRAIN_DIR_0,
    TRAIN_DIR_1,
    TRAIN_DIR_2,
    TRAIN_DIR_3,
    TRAIN_DIR_4,
    TRAIN_DIR_5,
    TRAIN_DIR_6,
    TRAIN_DIR_7,
    TRAIN_DIR_8,
    TRAIN_DIR_9,
    TRAIN_DIR_10,
    TRAIN_DIR_11,
    TRAIN_DIR_12,
    TRAIN_DIR_13,
    TRAIN_DIR_14,
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

with open(CHANS_LIST_layer5, "r") as file:
    channels_layer5 = file.read().splitlines()

with open(CHANS_LIST_layer6, "r") as file:
    channels_layer6 = file.read().splitlines()

with open(CHANS_LIST_layer7, "r") as file:
    channels_layer7 = file.read().splitlines()

with open(CHANS_LIST_layer8, "r") as file:
    channels_layer8 = file.read().splitlines()

with open(CHANS_LIST_layer9, "r") as file:
    channels_layer9 = file.read().splitlines()

with open(CHANS_LIST_layer10, "r") as file:
    channels_layer10 = file.read().splitlines()

with open(CHANS_LIST_layer11, "r") as file:
    channels_layer11 = file.read().splitlines()

with open(CHANS_LIST_layer12, "r") as file:
    channels_layer12 = file.read().splitlines()

with open(CHANS_LIST_layer13, "r") as file:
    channels_layer13 = file.read().splitlines()

with open(CHANS_LIST_layer14, "r") as file:
    channels_layer14 = file.read().splitlines()


TARGET_CHANNEL = "V1:Hrec_hoft_raw_20000Hz"
OUT_CHANNEL_layer0 = channels_layer1[0]
OUT_CHANNEL_layer1 = channels_layer2[0]
OUT_CHANNEL_layer2 = channels_layer3[0]
OUT_CHANNEL_layer3 = channels_layer4[0]
OUT_CHANNEL_layer4 = channels_layer5[0]
OUT_CHANNEL_layer5 = channels_layer6[0]
OUT_CHANNEL_layer6 = channels_layer7[0]
OUT_CHANNEL_layer7 = channels_layer8[0]
OUT_CHANNEL_layer8 = channels_layer9[0]
OUT_CHANNEL_layer9 = channels_layer10[0]
OUT_CHANNEL_layer10 = channels_layer11[0]
OUT_CHANNEL_layer11 = channels_layer12[0]
OUT_CHANNEL_layer12 = channels_layer13[0]
OUT_CHANNEL_layer13 = channels_layer14[0]
OUT_CHANNEL_FINAL = "V1:Hrec_hoft_raw_20000Hz_DC"

for CLEAN_T0 in range(1265172641, 1265229985, 4096):
    # =====================
    # Layer 0
    # =====================
    LAYER_NUM = "layer0"
    LOAD_DATASET = False
    FLOW = 142
    FHIGH = 162

    CHANS_LIST_layer = CHANS_LIST_layer0
    OUT_CHANNEL_layer = OUT_CHANNEL_layer0
    TRAIN_DIR = TRAIN_DIR_0

    # Layer 0 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device=DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 1
    # Dummy target will be replaced with the output of the previous layer 0
    # =====================================================================

    channels_with_dummy_target_layer1 = copy.deepcopy(channels_layer1)
    channels_with_dummy_target_layer1[0] = TARGET_CHANNEL

    # =====================
    # Layer 1
    # =====================
    LAYER_NUM = "layer1"
    LOAD_DATASET = True
    FLOW = 15
    FHIGH = 20
    TRAIN_DIR = TRAIN_DIR_1
    CHANS_LIST_layer = CHANS_LIST_layer1
    OUT_CHANNEL_layer = OUT_CHANNEL_layer1

    # Preparing data for Cleaning 1
    clean_t0 = CLEAN_T0
    data = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer1,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_0,
        new_target_channel=OUT_CHANNEL_layer0,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data.write(fname=os.path.join(TRAIN_DIR_1, new_frame), group=None, write_mode="w")

    # Layer 1 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device=DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 2
    # Dummy target will be replaced with the output of the previous layer 1
    # =====================================================================
    channels_with_dummy_target_layer2 = copy.deepcopy(channels_layer2)
    channels_with_dummy_target_layer2[0] = TARGET_CHANNEL

    # =====================
    # Layer 2
    # =====================
    LAYER_NUM = "layer2"
    LOAD_DATASET = True
    FLOW = 33
    FHIGH = 39
    TRAIN_DIR = TRAIN_DIR_2
    CHANS_LIST_layer = CHANS_LIST_layer2
    OUT_CHANNEL_layer = OUT_CHANNEL_layer2

    # Preparing data for Cleaning 2
    clean_t0 = CLEAN_T0
    data = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer2,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_1,
        new_target_channel=OUT_CHANNEL_layer1,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data.write(fname=os.path.join(TRAIN_DIR_2, new_frame), group=None, write_mode="w")

    # Layer 2 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device=DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 3
    # Dummy target will be replaced with the output of the previous layer 2
    # =====================================================================

    channels_with_dummy_target_layer3 = copy.deepcopy(channels_layer3)
    channels_with_dummy_target_layer3[0] = TARGET_CHANNEL

    # =====================
    # Layer 3
    # =====================
    LAYER_NUM = "layer3"
    LOAD_DATASET = True
    FLOW = 55
    FHIGH = 65
    TRAIN_DIR = TRAIN_DIR_3
    CHANS_LIST_layer = CHANS_LIST_layer3
    OUT_CHANNEL_layer = OUT_CHANNEL_layer3

    # Preparing data for Cleaning  for layer3
    clean_t0 = CLEAN_T0
    data = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer3,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_2,
        new_target_channel=OUT_CHANNEL_layer2,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data.write(fname=os.path.join(TRAIN_DIR_3, new_frame), group=None, write_mode="w")

    # Layer 3 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device="cpu",  # DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 4
    # Dummy target will be replaced with the output of the previous layer 3
    # =====================================================================

    channels_with_dummy_target_layer4 = copy.deepcopy(channels_layer4)
    channels_with_dummy_target_layer4[0] = TARGET_CHANNEL

    # =====================
    # Layer 4
    # =====================
    LAYER_NUM = "layer4"
    LOAD_DATASET = True
    FLOW = 75
    FHIGH = 80
    TRAIN_DIR = TRAIN_DIR_4
    CHANS_LIST_layer = CHANS_LIST_layer4
    OUT_CHANNEL_layer = OUT_CHANNEL_layer4

    # Preparing data for Cleaning  for layer4
    clean_t0 = CLEAN_T0
    data = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer4,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_3,
        new_target_channel=OUT_CHANNEL_layer3,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data.write(fname=os.path.join(TRAIN_DIR_4, new_frame), group=None, write_mode="w")

    # Layer 4 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device=DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 5
    # Dummy target will be replaced with the output of the previous layer 4
    # =====================================================================

    channels_with_dummy_target_layer5 = copy.deepcopy(channels_layer5)
    channels_with_dummy_target_layer5[0] = TARGET_CHANNEL

    # =====================
    # Layer 5
    # =====================
    LAYER_NUM = "layer5"
    LOAD_DATASET = True
    FLOW = 98
    FHIGH = 110
    TRAIN_DIR = TRAIN_DIR_5
    CHANS_LIST_layer = CHANS_LIST_layer5
    OUT_CHANNEL_layer = OUT_CHANNEL_layer5

    # Preparing data for Cleaning  for layer 5
    clean_t0 = CLEAN_T0
    data = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer5,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_4,
        new_target_channel=OUT_CHANNEL_layer4,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data.write(fname=os.path.join(TRAIN_DIR_5, new_frame), group=None, write_mode="w")

    # Layer 5 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device=DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 6
    # Dummy target will be replaced with the output of the previous layer 5
    # =====================================================================

    channels_with_dummy_target_layer6 = copy.deepcopy(channels_layer6)
    channels_with_dummy_target_layer6[0] = TARGET_CHANNEL

    # =====================
    # Layer 6
    # =====================
    LAYER_NUM = "layer6"
    LOAD_DATASET = True
    FLOW = 137
    FHIGH = 139
    TRAIN_DIR = TRAIN_DIR_6
    CHANS_LIST_layer = CHANS_LIST_layer6
    OUT_CHANNEL_layer = OUT_CHANNEL_layer6

    # Preparing data for Cleaning  for layer 6
    clean_t0 = CLEAN_T0
    data = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer6,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_5,
        new_target_channel=OUT_CHANNEL_layer5,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data.write(fname=os.path.join(TRAIN_DIR_6, new_frame), group=None, write_mode="w")

    # Layer 6 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device=DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 7
    # Dummy target will be replaced with the output of the previous layer6
    # =====================================================================

    channels_with_dummy_target_layer7 = copy.deepcopy(channels_layer7)
    channels_with_dummy_target_layer7[0] = TARGET_CHANNEL

    # =====================
    # Layer 7 Training and Cleaning
    # =====================
    LAYER_NUM = "layer7"
    LOAD_DATASET = True
    FLOW = 197
    FHIGH = 208
    TRAIN_DIR = TRAIN_DIR_7
    CHANS_LIST_layer = CHANS_LIST_layer7
    OUT_CHANNEL_layer = OUT_CHANNEL_layer7

    # Preparing data for Cleaning  for layer 7
    clean_t0 = CLEAN_T0
    data = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer7,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_6,
        new_target_channel=OUT_CHANNEL_layer6,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data.write(fname=os.path.join(TRAIN_DIR_7, new_frame), group=None, write_mode="w")

    # Layer 7 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device=DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 8
    # Dummy target will be replaced with the output of the previous layer 7
    # =====================================================================

    channels_with_dummy_target_layer8 = copy.deepcopy(channels_layer8)
    channels_with_dummy_target_layer8[0] = TARGET_CHANNEL

    # =====================
    # Layer 8 Training and Cleaning
    # =====================
    LAYER_NUM = "layer8"
    LOAD_DATASET = True
    FLOW = 247
    FHIGH = 252
    TRAIN_DIR = TRAIN_DIR_8
    CHANS_LIST_layer = CHANS_LIST_layer8
    OUT_CHANNEL_layer = OUT_CHANNEL_layer8

    # # Preparing data for Cleaning  for layer 8
    clean_t0 = CLEAN_T0
    data = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer8,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_7,
        new_target_channel=OUT_CHANNEL_layer7,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data.write(fname=os.path.join(TRAIN_DIR_8, new_frame), group=None, write_mode="w")

    # Layer 8 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device="cpu",  # DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 9
    # Dummy target will be replaced with the output of the previous layer 8
    # =====================================================================
    channels_with_dummy_target_layer9 = copy.deepcopy(channels_layer9)
    channels_with_dummy_target_layer9[0] = TARGET_CHANNEL

    # =====================
    # Layer 9 Training and Cleaning
    # =====================
    LAYER_NUM = "layer9"
    LOAD_DATASET = True
    FLOW = 295
    FHIGH = 305
    TRAIN_DIR = TRAIN_DIR_9
    CHANS_LIST_layer = CHANS_LIST_layer9
    OUT_CHANNEL_layer = OUT_CHANNEL_layer9

    # Preparing data for Cleaning for Layer 9
    clean_t0 = CLEAN_T0
    data_clean_layer9 = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer9,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_8,
        new_target_channel=OUT_CHANNEL_layer8,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data_clean_layer9.write(
        fname=os.path.join(TRAIN_DIR_9, new_frame), group=None, write_mode="w"
    )

    # Layer 9 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device="cpu",  # DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 10
    # Dummy target will be replaced with the output of the previous layer 9
    # =====================================================================

    channels_with_dummy_target_layer10 = copy.deepcopy(channels_layer10)
    channels_with_dummy_target_layer10[0] = TARGET_CHANNEL

    # =====================
    # Layer 10 Training and Cleaning
    # =====================

    LAYER_NUM = "layer10"
    LOAD_DATASET = True
    FLOW = 345
    FHIGH = 355
    TRAIN_DIR = TRAIN_DIR_10
    CHANS_LIST_layer = CHANS_LIST_layer10
    OUT_CHANNEL_layer = OUT_CHANNEL_layer10

    # Preparing data for Cleaning for Layer 10
    clean_t0 = CLEAN_T0
    data_clean_layer10 = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer10,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_9,
        new_target_channel=OUT_CHANNEL_layer9,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data_clean_layer10.write(
        fname=os.path.join(TRAIN_DIR_10, new_frame), group=None, write_mode="w"
    )

    # Layer 10 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device="cpu",  # DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 11
    # Dummy target will be replaced with the output of the previous layer 10
    # =====================================================================

    channels_with_dummy_target_layer11 = copy.deepcopy(channels_layer11)
    channels_with_dummy_target_layer11[0] = TARGET_CHANNEL

    # =====================
    # Layer 11
    # =====================
    LAYER_NUM = "layer11"
    LOAD_DATASET = True
    FLOW = 355
    FHIGH = 367
    TRAIN_DIR = TRAIN_DIR_11
    CHANS_LIST_layer = CHANS_LIST_layer11
    OUT_CHANNEL_layer = OUT_CHANNEL_layer11

    # Preparing data for Cleaning  for layer11
    clean_t0 = CLEAN_T0
    data = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer11,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_10,
        new_target_channel=OUT_CHANNEL_layer10,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data.write(fname=os.path.join(TRAIN_DIR_11, new_frame), group=None, write_mode="w")

    # Layer 11 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device=DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 12
    # Dummy target will be replaced with the output of the previous layer 11
    # =====================================================================

    channels_with_dummy_target_layer12 = copy.deepcopy(channels_layer12)
    channels_with_dummy_target_layer12[0] = TARGET_CHANNEL

    # =====================
    # Layer 12
    # =====================
    LAYER_NUM = "layer12"
    LOAD_DATASET = True
    FLOW = 395
    FHIGH = 415
    TRAIN_DIR = TRAIN_DIR_12
    CHANS_LIST_layer = CHANS_LIST_layer12
    OUT_CHANNEL_layer = OUT_CHANNEL_layer12

    # Preparing data for Cleaning  for layer12
    clean_t0 = CLEAN_T0
    data = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer12,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_11,
        new_target_channel=OUT_CHANNEL_layer11,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data.write(fname=os.path.join(TRAIN_DIR_12, new_frame), group=None, write_mode="w")

    # Layer 12 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device=DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 13
    # Dummy target will be replaced with the output of the previous layer 12
    # =====================================================================

    channels_with_dummy_target_layer13 = copy.deepcopy(channels_layer13)
    channels_with_dummy_target_layer13[0] = TARGET_CHANNEL

    # =====================
    # Layer 13
    # =====================
    LAYER_NUM = "layer13"
    LOAD_DATASET = True
    FLOW = 548
    FHIGH = 555
    TRAIN_DIR = TRAIN_DIR_13
    CHANS_LIST_layer = CHANS_LIST_layer13
    OUT_CHANNEL_layer = OUT_CHANNEL_layer13

    # Preparing data for Cleaning  for layer13
    clean_t0 = CLEAN_T0
    data = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer13,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_12,
        new_target_channel=OUT_CHANNEL_layer12,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data.write(fname=os.path.join(TRAIN_DIR_13, new_frame), group=None, write_mode="w")

    # Layer 13 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device=DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )

    # =====================================================================
    # Prepare for Layer 14
    # Dummy target will be replaced with the output of the previous layer 13
    # =====================================================================

    channels_with_dummy_target_layer14 = copy.deepcopy(channels_layer14)
    channels_with_dummy_target_layer14[0] = TARGET_CHANNEL

    # =====================
    # Layer 14
    # =====================
    LAYER_NUM = "layer14"
    LOAD_DATASET = True
    FLOW = 598
    FHIGH = 603
    TRAIN_DIR = TRAIN_DIR_14
    CHANS_LIST_layer = CHANS_LIST_layer14
    OUT_CHANNEL_layer = OUT_CHANNEL_FINAL

    # Preparing data for Cleaning  for layer14
    clean_t0 = CLEAN_T0
    data = get_local_data_and_replace_target(
        cache_file=CACHE_FILE,
        channels=channels_with_dummy_target_layer14,
        t0=clean_t0,
        duration=CLEAN_DURATION,
        new_target_read_from=TRAIN_DIR_13,
        new_target_channel=OUT_CHANNEL_layer13,
        fs=FS,
    )
    new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
    data.write(fname=os.path.join(TRAIN_DIR_14, new_frame), group=None, write_mode="w")

    # Layer 14 Cleaning
    print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
    clean_layer(
        cache_file=CACHE_FILE,
        chans_list_layer=CHANS_LIST_layer,
        train_dir=TRAIN_DIR,
        clean_t0=CLEAN_T0,
        clean_duration=CLEAN_DURATION,
        out_channel_layer=OUT_CHANNEL_layer,
        layer_num=LAYER_NUM,
        load_dataset=LOAD_DATASET,
        fs=FS,
        device=DEVICE,
        ifo=IFO,
        prefix=PREFIX,
    )
