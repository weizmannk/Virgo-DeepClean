
# dc-prod-clean  --load-dataset True --ifo V1 --save-dataset True --fs 4096 --out-dir 15-415_Hz/StainGWsignals/layer0  --out-channel V1:Hrec_hoft_raw_20000Hz_DC --chanslist  /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/witnesses/witnesses_142-162_Hz.ini   --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device cuda --train-dir /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/4096/layer0/Hrec-HOFT-1265127585-100000  --clean-t0 1265127585 --clean-duration 4096 --out-file Hrec-HOFT-1265127585-4096.gwf


#dc-prod-clean  --load-dataset True --ifo V1 --save-dataset True --fs 4096 --out-dir 142-162_5_Mpc/ORG_NoisePlusSignal  --out-channel V1:Hrec_hoft_raw_20000Hz_DC --chanslist  /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Injections/witnesses/witnesses_142-162_Hz.ini    --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device cuda --train-dir /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/142-162_Hz/4096/Hrec-HOFT-1265127585-50000/train --clean-t0 1265127585 --clean-duration 4096 --out-file Hrec-HOFT-1265127585-4096.gwf

# #!/bin/bash

# # Define start and end times for each pair
# s1=1265127585; t1=1265225889
# s2=1265227585; t2=1265325889
# s3=1265327585; t3=1265352161

# # Duration
# duration=4096

# # Define an array of pairs
# time_pairs=("$s1 $t1" "$s2 $t2" "$s3 $t3")



s1=1265127585; t1=1265172641
s2=1265176737; t2=1265225889
s3=1265227585; t3=1265272641
s4=1265276737; t4=1265325889
s5=1265327585; t5=1265352161

# Duration
duration=4096

# Define an array of pairs
time_pairs=("$s1 $t1" "$s2 $t2" "$s3 $t3" "$s4 $t4" "$s5 $t5")

# Loop over each (initial_time, end_time) pair
for pair in "${time_pairs[@]}"; do
    # Read initial_time and end_time from the pair
    read initial_time end_time <<< "$pair"


    cadence=50000
#     if [ "$initial_time" -eq 1265327585 ]; then
#        cadence=38534
#     fi

    # Reset start_time to initial_time at the beginning of each pair's loop
    start_time=$initial_time

    while [ "$(($start_time ))" -le "$end_time" ]; do

        dc-prod-clean --load-dataset True --ifo V1 --save-dataset True --chunk-duration None --fs 4096 \
        --out-dir Rework/142-162_Hz/noisePlusSignal --out-channel V1:Hrec_hoft_raw_20000Hz_DC \
        --chanslist /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/retrain/witnesses/witnesses_142-162_Hz.ini   \
        --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device cpu \
        --train-dir /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/142-162_Hz/4096/Hrec-HOFT-${initial_time}-${cadence}\
        --clean-t0 $start_time --clean-duration 4096 \
        --out-file Hrec-HOFT-${start_time}-${duration}.gwf

        # Update start_time for the next iteration
        start_time=$(($start_time + $duration))
    done
done





# dc-prod-clean --load-dataset True --ifo V1 --save-dataset True --chunk-duration None --fs 4096 --out-dir Rework/142-162_Hz/noisePlusSignal --out-channel V1:Hrec_hoft_raw_20000Hz_DC --chanslist  /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/retrain/witnesses/witnesses_142-162_Hz.ini   --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device cpu --train-dir /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/142_162_Hz/4096/Hrec-HOFT-1265127585-100000/ --clean-t0 1265127585 --clean-duration 4096 --out-file HOFT-1265127585-4096.gwf


# dc-prod-clean --load-dataset True --ifo V1 --save-dataset True --chunk-duration None --fs 4096 --out-dir 15-415_Hz/noisePlusSignal --out-channel V1:Hrec_hoft_raw_20000Hz_DC --chanslist /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/retrain/witnesses/witnesses_142-162_Hz.ini --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device cpu --train-dir /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/retrain/4096/layer0/Hrec-HOFT-1265127585-100000 --clean-t0 1265127585 --clean-duration 4096 --out-file HOFT-1265127585-4096_layer1.gwf


# dc-prod-clean --load-dataset True --ifo V1 --save-dataset True --chunk-duration None --fs 4096 --out-dir rework_15_415_Hz/noisePlusSignal/layer0 --out-channel V1:DC_layer0--chanslist /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/witnesses/witnesses_142-162_Hz.ini --clean-kernel 8 --clean-stride 4 --pad-mode median --window hanning --device cpu --train-dir /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/4096/layer0/Hrec-HOFT-1265127585-100000 --clean-t0 1265127585 --clean-duration 4096 --out-file HOFT-1265127585-4096_layer0.gwf
