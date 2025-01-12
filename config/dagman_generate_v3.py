import os
import sys
import argparse
import subprocess
import random
import secrets

from subprocess import check_output

import numpy as np
from deepclean_prod import io

# Parse command line argument
def parse_cmd():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), usage="%(prog)s [options]"
    )
    parser.add_argument("config", help="Path to config file", type=str)
    parser.add_argument("--submit", help="Submit DAGMAN", action="store_true")
    params = parser.parse_args()
    return params


params = parse_cmd()

# Parse config file
config = io.parse_config(params.config, "config")
out_dir = config["out_dir"]
prefix = config.get("prefix", "prefix")
job_name = config.get("job_name", "job")
# name extended by a random number, to avoid repeataion
job_name_extended = job_name + "_" + secrets.token_hex(8)
max_clean_duration = int(config.get("max_clean_duration", 4096))

request_memory_training = config.get("request_memory_training", "2 GB")
request_memory_cleaning = config.get("request_memory_cleaning", "2 GB")
# request_disk            = config.get('request_disk', '1 GB')


# dcprodtrain =  config.get('dcprodtrain', check_output(["which", "dc-prod-train"]).decode().replace("\n", ""))
# dcprodtrain =  config.get(''dcprodclean', check_output(["which", "dc-prod-train"]).decode().replace("\n", ""))

dcprodtrain = config.get(
    "dcprodtrain", os.popen("which dc-prod-train").read().strip("\n")
)
dcprodclean = config.get(
    "dcprodclean", os.popen("which dc-prod-clean").read().strip("\n")
)


# Create output directory
out_dir_log = os.path.join(out_dir, "condor_logs")
out_dir_submit = os.path.join(out_dir, "condor", "submit")
# out_dir_submit = './'

os.makedirs(out_dir_log, exist_ok=True)
os.makedirs(out_dir_submit, exist_ok=True)

# request_disk   = '''+request_disk+'''

## train.sub content

# request_disk   = '''+request_disk+'''
train_sub = (
    """
universe = vanilla
executable = """
    + dcprodtrain
    + """
request_memory = """
    + request_memory_training
    + """
getenv = True
requirements = (CUDACapability > 3.5)
log    = """
    + out_dir
    + """/condor_logs/train_$(ID).log
output = """
    + out_dir
    + """/condor_logs/train_$(ID).out
error  = """
    + out_dir
    + """/condor_logs/train_$(ID).err
accounting_group =  ligo.dev.o4.detchar.subtraction.deepclean ##ligo.prod.o3.detchar.nonlin_coup.bcv
request_gpus = 1
stream_output = True
stream_error = True
arguments = $(ARGS)
queue
"""
)
# request_disk   = '''+request_disk+'''
clean_sub = (
    """
universe = vanilla
executable = """
    + dcprodclean
    + """
request_memory = """
    + request_memory_cleaning
    + """
getenv = True
requirements = (CUDACapability > 3.5)
log    = """
    + out_dir
    + """/condor_logs/clean_$(ID).log
output = """
    + out_dir
    + """/condor_logs/clean_$(ID).out
error  = """
    + out_dir
    + """/condor_logs/clean_$(ID).err
accounting_group = ligo.dev.o4.detchar.subtraction.deepclean ##ligo.prod.o3.detchar.nonlin_coup.bcv
request_gpus = 1
stream_output = True
stream_error = True
arguments = $(ARGS)
queue
"""
)


def get_train_ARGS(config, train_duration):

    arg = ""
    arg += " --save-dataset "
    arg += config["save_dataset"]

    # arg += " --load-dataset "
    # arg += config['load_dataset']

    arg += " --fs "
    arg += config["fs"]

    arg += " --chanslist "
    arg += config["chanslist"]

    arg += " --train-kernel "
    arg += config["train_kernel"]

    arg += " --train-stride "
    arg += config["train_stride"]

    arg += " --pad-mode "
    arg += config["pad_mode"]

    arg += " --filt-fl "
    arg += config["filt_fl"]

    arg += " --filt-fh "
    arg += config["filt_fh"]

    arg += " --filt-order "
    arg += config["filt_order"]

    arg += " --device "
    arg += config["device"]

    arg += " --train-frac "
    arg += config["train_frac"]

    arg += " --batch-size "
    arg += config["batch_size"]

    arg += " --max-epochs "
    arg += config["max_epochs"]

    arg += " --num-workers "
    arg += config["num_workers"]

    arg += " --lr "
    arg += config["lr"]

    arg += " --weight-decay "
    arg += config["weight_decay"]

    arg += " --fftlength "
    arg += config["fftlength"]

    arg += " --psd-weight "
    arg += config["psd_weight"]

    arg += " --mse-weight "
    arg += config["mse_weight"]

    arg += " --train-dir "
    arg += train_config["train_dir"]

    arg += " --train-t0 "
    arg += str(train_config["train_t0"])

    arg += " --train-duration "
    arg += str(train_config["train_duration"])

    return arg


def get_clean_ARGS(config, clean_config):

    arg = ""
    arg += " --save-dataset "
    arg += config["save_dataset"]

    # arg += " --load-dataset "
    # arg += config['load_dataset']

    arg += " --fs "
    arg += config["fs"]

    arg += " --out-dir "
    arg += clean_config["out_dir"]

    arg += " --out-channel "
    arg += config["out_channel"]

    arg += " --chanslist "
    arg += config["chanslist"]

    arg += " --clean-kernel "
    arg += config["clean_kernel"]

    arg += " --clean-stride "
    arg += config["clean_stride"]

    arg += " --pad-mode "
    arg += config["pad_mode"]

    arg += " --window "
    arg += config["window"]

    arg += " --device "
    arg += config["device"]

    arg += " --train-dir "
    arg += clean_config["train_dir"]

    arg += " --clean-t0 "
    arg += str(clean_config["clean_t0"])

    arg += " --clean-duration "
    arg += str(clean_config["clean_duration"])

    arg += " --out-file "
    arg += clean_config["out_file"]

    return arg


# Run segment script
config["segment_file"] = os.path.join(out_dir, "segment.txt")
segment_cmd = "dc-prod-segment " + io.dict2args(
    config,
    (
        "t0",
        "t1",
        "train_duration",
        "train_cadence",
        "start_training_after",
        "ifo",
        "segment_file",
    ),
)
print("Get segment data")
subprocess.check_call(segment_cmd.split(" "))

# Read in segment data
segment_data = np.genfromtxt(config["segment_file"])

dag_script = """"""
interjob_dependence = """"""

dag_sh = """"""

train_submit_filename = "deepClean_train_{}.sub".format(job_name)
clean_submit_filename = "deepClean_clean_{}.sub".format(job_name)
dag_submit_filename = "deepClean_dag_{}.dag".format(job_name)
dag_sh_filename = "deepClean_dag_{}.sh".format(job_name)

# dag_id will be used as extentions for job names within the dag script
dag_id = 0

for seg in segment_data:

    # Get training/cleaning time
    train_t0, train_t1, clean_t0, clean_t1, chunk_idx = seg.astype(int)
    train_duration = train_t1 - train_t0
    clean_duration = clean_t1 - clean_t0

    # Get directory for segment
    segment_subdir = os.path.join(
        out_dir, "{}-{:d}-{:d}".format(prefix, clean_t0, clean_duration)
    )

    # job identifiers within the dag script
    train_job = "train_" + job_name_extended + "_" + str(dag_id)
    clean_job = "clean_" + job_name_extended + "_" + str(dag_id)

    # Set up training job
    train_config = {}
    train_config["train_dir"] = segment_subdir
    train_config["train_t0"] = train_t0
    train_config["train_duration"] = train_duration

    # generate argument string for training job
    train_arg = get_train_ARGS(config, train_config)
    dc_prod_train_sh = "dc-prod-train {}\n".format(train_arg)

    # dag entry for the training job
    cmd_job_train = """JOB {} {}""".format(
        train_job, os.path.join(out_dir_submit, train_submit_filename)
    )
    cmd_retry_train = """RETRY {} {:d}""".format(train_job, 10)
    cmd_var_train = """VARS {} ARGS="{}" ID="{:d}" """.format(
        train_job, train_arg, dag_id
    )

    dag_script += "\n\n#Jobs for data between GPS times {:d} and {:d}\n".format(
        clean_t0, clean_t1
    )
    dag_script += (
        "#----------------------------------------------------------------------\n"
    )
    dag_script += cmd_job_train + "\n"
    dag_script += cmd_retry_train + "\n"
    dag_script += cmd_var_train + "\n\n"

    dag_sh += "\n\n#Executables for data between GPS times {:d} and {:d}\n".format(
        clean_t0, clean_t1
    )
    dag_sh += (
        "#----------------------------------------------------------------------\n"
    )
    dag_sh += dc_prod_train_sh + "\n"

    # Set up cleaning jobs (could be more than 1)
    clean_config = {}
    clean_config["train_dir"] = segment_subdir
    clean_config["out_dir"] = segment_subdir

    # if the active segment length > max_clean_duration, then split as multiple segments
    t0 = np.arange(clean_t0, clean_t1, max_clean_duration)
    for i in range(len(t0)):

        # job name further extended for <i>th clean job parented by <dag_id>th train job
        clean_job_i = "{}_{:d}".format(clean_job, i)
        #
        clean_config["clean_t0"] = t0[i]
        if t0[i] + max_clean_duration > clean_t1:
            duration = clean_t1 - t0[i]
        else:
            duration = max_clean_duration
        clean_config["clean_duration"] = duration
        clean_config["out_file"] = "{}-{:d}-{:d}.gwf".format(prefix, t0[i], duration)

        # generate argument string for cleaning job
        clean_arg = get_clean_ARGS(config, clean_config)
        dc_prod_clean_sh = "dc-prod-clean {}\n".format(clean_arg)

        cmd_job_clean = """JOB {} {}""".format(
            clean_job_i, os.path.join(out_dir_submit, clean_submit_filename)
        )
        cmd_retry_clean = """RETRY {} {:d}""".format(clean_job_i, 5)

        cmd_var_clean = """VARS {} ARGS="{}" ID="{:d}_{:d}" """.format(
            clean_job_i, clean_arg, dag_id, i
        )

        dag_script += cmd_job_clean + "\n"
        dag_script += cmd_retry_clean + "\n"
        dag_script += cmd_var_clean + "\n\n"

        dag_sh += dc_prod_clean_sh + "\n"

        interjob_dependence += "Parent " + train_job + " Child " + clean_job_i + "\n"

    dag_id += 1


dag_script += "\n\n"
dag_script += interjob_dependence

# write submit files
text_file = open(os.path.join(out_dir_submit, train_submit_filename), "w")
text_file.write(train_sub)
text_file.close()

text_file = open(os.path.join(out_dir_submit, clean_submit_filename), "w")
text_file.write(clean_sub)
text_file.close()

text_file = open(os.path.join(out_dir_submit, dag_submit_filename), "w")
text_file.write(dag_script)
text_file.close()

text_file = open(os.path.join(out_dir_submit, dag_sh_filename), "w")
text_file.write(dag_sh)
text_file.close()

cmd_dag_submit = "condor_submit_dag {}".format(
    os.path.join(os.getcwd(), out_dir_submit, dag_submit_filename)
)

if params.submit:
    os.system(cmd_dag_submit)
else:
    print("\nDag generated successfully . .\n")
    print(
        "To submit the dag, run the following command from terminal:\n\t{}\n".format(
            cmd_dag_submit
        )
    )
