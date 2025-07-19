#!/usr/bin/env python

"""
---------------------------------------------------------------------------------------------------
ABOUT THE SCRIPT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository URL  : https://github.com/weizmannk/Virgo-DeepClean.git
Creation Date   : Jun 2025
Description     : 


"""

import os

# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=1

# CUDA device order and visible devices
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


def get_local_data_and_replace_target(cache_file, channels, t0, duration, new_target_read_from, new_target_channel="V1:Hrec_hoft_raw_20000Hz_DC", fs=4096.0):
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
    new_target_files = glob.glob(f"{new_target_read_from}/*.gwf")
    new_data = TimeSeries.read(new_target_files, channel=new_target_channel, start=t0, end=t0 + duration)
    data.data[data.target_idx] = new_data
    data.channels[data.target_idx] = new_target_channel
    return data



def execute_command(command):
    """
    Executes a given shell command and prints the output line by line.

    Parameters:
    - command (str): The command to execute.
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        output = process.stdout.readline()
        if not output and process.poll() is not None:
            break
        if output:
            print(output.strip().decode())

    rc = process.poll()
    return rc


def train_layer(cache_file, chans_list_layer, flow, fhigh, train_t0, train_duration, train_dir, load_dataset=False, train_fac=0.9, fs=4096, device ="cuda", ifo="V1"):
    train_command_layer = f"dc-prod-train  --cache-file {cache_file} --ifo {ifo} --load-dataset {load_dataset} --save-dataset True --fs {fs} --chanslist {chans_list_layer} --train-kernel 8 --train-stride 0.25 --pad-mode median --filt-fl {flow} --filt-fh {fhigh} --filt-order 8 --device {device} --train-frac {train_fac} --batch-size 32 --max-epochs 30 --num-workers 4 --lr 1e-3 --weight-decay 1e-5 --fftlength 2 --psd-weight 1.0 --mse-weight 0.0 --train-dir {train_dir} --train-t0 {train_t0} --train-duration {train_duration}"
    
    
    execute_command(train_command_layer)
    #print(train_command_layer)

def clean_layer(cache_file, chans_list_layer, train_dir, clean_t0, clean_duration, out_channel_layer, layer_num, load_dataset=False, fs=4096, device="cuda", ifo="V1", prefix="Hrec-HOFT"):
    clean_command_layer = f"dc-prod-clean --cache-file {cache_file} --ifo {ifo} --load-dataset {load_dataset} --save-dataset True --fs {fs} --out-dir {train_dir} --out-channel {out_channel_layer} --chanslist {chans_list_layer} --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device {device} --train-dir {train_dir} --clean-t0 {clean_t0} --clean-duration {clean_duration} --out-file {prefix}-{clean_t0}-{clean_duration}_{layer_num}.gwf"
    
    execute_command(clean_command_layer)
    #print(clean_command_layer)


CACHE_FILE = "/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/V1-O3b_1265100000-1265399999.cache"


# --------- PARAMETERS ---------
NUM_LAYERS = 122  # from 0 to 121
IFO = "V1"
DEVICE = "cuda"
FS = 4096
TRAIN_T0 =   1265127585
TRAIN_DURATION = 4096
CLEAN_T0 = 1265127585
CLEAN_DURATION = 4096
TRAIN_FAC = 0.9

# Frequences min and max
FLOW  = 142
FHIGH = 162

PREFIX = "Hrec-HOFT"
OUT_DIR = "4096_sequential" 
TRAIN_CADENCE = "100000"

CACHE_FILE = "/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/V1-O3b_1265100000-1265399999.cache"
TARGET_CHANNEL = "V1:Hrec_hoft_raw_20000Hz"

# Optionally, define the "final" output channel
OUT_CHANNEL_FINAL = "V1:Hrec_hoft_raw_20000Hz_DC"

# --------- CREATE TRAINING DIRECTORIES ---------
train_dirs = []
for i in range(NUM_LAYERS):
    train_dir = f"{OUT_DIR}/layer{i}/{PREFIX}-{TRAIN_T0}-{TRAIN_CADENCE}"
    train_dirs.append(train_dir)
    os.makedirs(train_dir, exist_ok=True)

# --------- GENERATE CHANNELS LIST PATHS ---------
chans_list_files = [f"./witnesses-sequential/layer{i}.ini" for i in range(NUM_LAYERS)]

# --------- LOAD CHANNELS FOR EACH LAYER ---------
channels_per_layer = []
for chans_file in chans_list_files:
    with open(chans_file, 'r') as file:
        chans = [l.strip() for l in file if l.strip() and not l.startswith('[')]
        # Only keep non-empty, non-section-header lines
        # If ini has "channel = X", keep the right-hand side
        chans = [line.split('=', 1)[1].strip() if '=' in line else line for line in chans]
        channels_per_layer.append(chans)


# --------- DEFINE OUT_CHANNEL FOR EACH LAYER (from 1st to last-1) ---------
# Convention: OUT_CHANNEL_layerN = first channel of layer N+1)
out_channels = [channels_per_layer[i][0] for i in range(1, NUM_LAYERS)]
out_channels.append(OUT_CHANNEL_FINAL)


# --- MAIN LAYER LOOP ---
for layer_idx in range(NUM_LAYERS)[2:]:
    print(f"\n---- Layer {layer_idx} ----")
    # Set key variables for this layer
    TRAIN_DIR = train_dirs[layer_idx]
    CHANS_LIST_layer = chans_list_files[layer_idx]
    OUT_CHANNEL_layer = out_channels[layer_idx]

    LAYER_NUM = f"layer{layer_idx}"

    if layer_idx == 0:
        print(f"Training for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
        train_layer(
            cache_file=CACHE_FILE, 
            chans_list_layer=CHANS_LIST_layer, 
            flow=FLOW, 
            fhigh=FHIGH, 
            train_t0=TRAIN_T0, 
            train_duration=TRAIN_DURATION, 
            train_dir=TRAIN_DIR, 
            load_dataset=False, 
            train_fac=TRAIN_FAC, 
            fs=FS, 
            device=DEVICE, 
            ifo=IFO
        )
        print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
        clean_layer(
            cache_file=CACHE_FILE, 
            chans_list_layer=CHANS_LIST_layer,
            train_dir=TRAIN_DIR, 
            clean_t0=CLEAN_T0, 
            clean_duration=CLEAN_DURATION, 
            out_channel_layer=OUT_CHANNEL_layer, 
            layer_num=LAYER_NUM, 
            load_dataset=False, 
            fs=FS, 
            device=DEVICE,
            ifo=IFO, 
            prefix=PREFIX
        )

    else:

        # --- Prepare channels for get_local_data_and_replace_target
        channels_with_dummy_target = copy.deepcopy(channels_per_layer[layer_idx])
        channels_with_dummy_target[0] = TARGET_CHANNEL

        # --- Data Preparation for Training and Validation
        train_length = int(TRAIN_DURATION * TRAIN_FAC)
        val_start = TRAIN_T0 + train_length
        val_length = TRAIN_DURATION - train_length

        prev_train_dir = train_dirs[layer_idx - 1] if layer_idx > 0 else train_dirs[0]
        prev_out_channel = out_channels[layer_idx - 1] if layer_idx > 0 else out_channels[0]

        # Training set
        data_train = get_local_data_and_replace_target(
            cache_file=CACHE_FILE, 
            channels=channels_with_dummy_target, 
            t0=TRAIN_T0, 
            duration=train_length, 
            new_target_read_from=prev_train_dir, 
            new_target_channel=prev_out_channel, 
            fs=FS
        )
        data_train.write(fname=os.path.join(TRAIN_DIR, 'training.h5'), group=None, write_mode='w')

        # Validation set
        data_val = get_local_data_and_replace_target(
            cache_file=CACHE_FILE, 
            channels=channels_with_dummy_target, 
            t0=val_start, 
            duration=val_length, 
            new_target_read_from=prev_train_dir, 
            new_target_channel=prev_out_channel, 
            fs=FS
        )
        data_val.write(fname=os.path.join(TRAIN_DIR, 'validation.h5'), group=None, write_mode='w')
       
        # --- Training ---
        print(f"Training for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
        train_layer(
            cache_file=CACHE_FILE, 
            chans_list_layer=CHANS_LIST_layer, 
            flow=FLOW, 
            fhigh=FHIGH, 
            train_t0=TRAIN_T0, 
            train_duration=TRAIN_DURATION, 
            train_dir=TRAIN_DIR, 
            load_dataset=(layer_idx > 0), 
            train_fac=TRAIN_FAC, 
            fs=FS, 
            device=DEVICE, 
            ifo=IFO
        )
        
        # --- Cleaning Preparation ---
        clean_t0 = CLEAN_T0
        data_clean = get_local_data_and_replace_target(
            cache_file=CACHE_FILE, 
            channels=channels_with_dummy_target, 
            t0=clean_t0, 
            duration=CLEAN_DURATION, 
            new_target_read_from=prev_train_dir, 
            new_target_channel=prev_out_channel,
            fs=FS
        )
        new_frame = f"original-{clean_t0}-{CLEAN_DURATION}.h5"
        data_clean.write(fname=os.path.join(TRAIN_DIR, new_frame), group=None, write_mode='w')

        # --- Cleaning ---
        print(f"Cleaning for {FLOW}-{FHIGH} Hz band {LAYER_NUM}...")
        clean_layer(
            cache_file=CACHE_FILE, 
            chans_list_layer=CHANS_LIST_layer,
            train_dir=TRAIN_DIR, 
            clean_t0=CLEAN_T0, 
            clean_duration=CLEAN_DURATION, 
            out_channel_layer=OUT_CHANNEL_layer, 
            layer_num=LAYER_NUM, 
            load_dataset=(layer_idx > 0), 
            fs=FS, 
            device=DEVICE,
            ifo=IFO, 
            prefix=PREFIX
        )
        