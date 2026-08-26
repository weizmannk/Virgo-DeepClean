import os
import textwrap
import subprocess
import logging

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


output = "98-110_Hz/matchedfilter_SNR_Freq"
os.makedirs(output, exist_ok=True)

log_dir = f"{output}/logs"
wrapper_dir = f"{output}/wrappers"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(wrapper_dir, exist_ok=True)

# Parametters
time_pairs = [
    (1265127585, 1265225889),
    (1265227585, 1265325889),
    (1265327585, 1265352161),
]
duration = 4096

job = 1
# Loop through the time pairs
for idx, (initial_time, end_time) in enumerate(time_pairs, start=1):
    start_time = initial_time
    while start_time + duration <= end_time:
        job_name = f"job_{job}_{start_time}"

        # Create the wrapper script
        wrapper_script = os.path.join(wrapper_dir, f"{job_name}.sh")
        wrapper_content = textwrap.dedent(
            f"""\
            #!/bin/bash
            python matched_filter_snr_freq.py --inj-gap 32 \\
                --channel-inj V1:DC_INJ \\
                --channel-DC V1:Hrec_hoft_raw_20000Hz_DC \\
                --channel-org V1:Hrec_hoft_raw_20000Hz \\
                --injections-file /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Injections/new_injection/CNR-version/special/Rework/98-110_Hz/injSignal/INJ-{start_time}-{duration}.csv \\
                --inj-gw-signal /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Injections/new_injection/CNR-version/special/Rework/98-110_Hz/injSignal/V1_BBH_inj_{start_time}_{duration}.hdf5 \\
                --org-noise /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/98-110_Hz/4096/Hrec-HOFT-{initial_time}-100000/original-{start_time}-{duration}.h5 \\
                --org-noise-gw-signal /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Injections/new_injection/CNR-version/special/Rework/98-110_Hz/noisePlusSignal/original-{start_time}-{duration}.h5 \\
                --cleaned-noise /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/98-110_Hz/4096/Hrec-HOFT-{initial_time}-100000/Hrec-HOFT-{start_time}-{duration}.gwf \\
                --cleaned-gw-signal /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Injections/new_injection/CNR-version/special/Rework/98-110_Hz/noisePlusSignal/Hrec-HOFT-{start_time}-{duration}.gwf \\
                --fmin 197 --fmax 208 \\
                --outdir {output}/4096_32s --verbose
        """
        )
        try:
            with open(wrapper_script, "w") as f:
                f.write(wrapper_content)
            os.chmod(wrapper_script, 0o755)
            logging.info(f"Created wrapper script: {wrapper_script}")
        except Exception as e:
            logging.error(f"Failed to create wrapper script for job {job_name}: {e}")
            continue

        # Create the Condor submission script without indentation
        condor_submit_script = textwrap.dedent(
            f"""\
            +MaxHours = 24
            universe = vanilla
            accounting_group = ligo.dev.o4.cbc.pe.bayestar
            getenv = true
            executable = {wrapper_script}
            output = {log_dir}/{job_name}_$(Cluster)_$(Process).out
            error = {log_dir}/{job_name}_$(Cluster)_$(Process).err
            log = {log_dir}/{job_name}_$(Cluster)_$(Process).log
            JobBatchName = DeepClean_{job_name}
            request_memory = 20 GB
            request_disk = 6 GB
            request_cpus = 1
            on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)
            on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
            on_exit_hold_reason = (ExitBySignal == True ? strcat("The job exited with signal ", ExitSignal) : strcat("The job exited with code ", ExitCode))
            environment = "OMP_NUM_THREADS=1"
            queue 1
        """
        )
        logging.debug(
            f"Condor submit script for job {job_name}:\n{condor_submit_script}"
        )

        # Submit the Condor job
        try:
            proc = subprocess.Popen(
                ["condor_submit"],
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate(input=condor_submit_script)
            if proc.returncode == 0:
                logging.info(
                    f"Condor submit output for {job_name}, job {job}: {stdout.strip()}"
                )
            else:
                logging.error(
                    f"Condor submit error for {job_name}, job {job}: {stderr.strip()}"
                )
        except Exception as e:
            logging.error(
                f"An error occurred while creating the Condor submission for {job_name}, job {job}: {e}"
            )

        # Update start_time for the next job
        start_time += duration
        job += 1
