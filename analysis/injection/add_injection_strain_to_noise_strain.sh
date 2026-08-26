# #/home/weizmann.kiendrebeogo/OBSERVING_SCENARIOS/HL_observing_SNR_10/runs



#!/bin/bash


# s1=1265127585; t1=1265172641
# s2=1265176737; t2=1265225889
# s3=1265227585; t3=1265272641
# s4=1265276737; t4=1265325889
# s5=1265327585; t5=1265352161

# # Duration
# duration=4096

# # Define an array of pairs
# time_pairs=("$s1 $t1" "$s2 $t2" "$s3 $t3" "$s4 $t4" "$s5 $t5")


# Define start and end times for each pair
s1=1265127585; t1=1265225889
s2=1265227585; t2=1265325889
s3=1265327585; t3=1265352161

# Duration
duration=4096


# Define an array of pairs
time_pairs=("$s1 $t1" "$s2 $t2" "$s3 $t3")

# # Define an array of pairs
# time_pairs=("$s1 $t1" "$s2 $t2")

# Loop over each (initial_time, end_time) pair
for pair in "${time_pairs[@]}"; do
    # Read initial_time and end_time from the pair
    read initial_time end_time <<< "$pair"


    cadence=100000
    if [ "$initial_time" -eq 1265327585 ]; then
       cadence=38534
    fi

    # Reset start_time to initial_time at the beginning of each pair's loop
    start_time=$initial_time

    while [ "$(($start_time))" -le "$end_time" ]; do
        # Construct the noise frame path using the calculated start time
        noise_frame="/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/197-208_Hz/4096/Hrec-HOFT-${initial_time}-${cadence}/original-${start_time}-4096.h5"

        # Construct the injection frame path
        inj_frame="/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Injections/new_injection/CNR-version/special/Rework/197-208_Hz/injSignal/V1_BBH_inj_${start_time}_4096.hdf5"

        # Output directory
        output_frame="/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Injections/new_injection/CNR-version/special/Rework/197-208_Hz/noisePlusSignal"

        # Echo command for verification or logging
        echo "Processing $noise_frame"


        # Run the Python script with constructed arguments
        python add_injection_strain_to_noise_strain.py --strain-channel V1:Hrec_hoft_raw_20000Hz --chanslist /home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/197-208_Hz/197-208_Hz.ini --noise-frame "$noise_frame" --inj-frame "$inj_frame" --output-frame "$output_frame"

        # Update start_time for the next iteration
        start_time=$(($start_time + $duration))
    done
done
echo "All injections added to noise strain."



# #!/bin/bash

# # Define start and end times for each pair
# s1=1265127585; t1=1265225889
# # Additional pairs can be uncommented and added here

# # Duration
# duration=4096

# # Define an array of pairs
# time_pairs=("$s1 $t1") # Add more pairs as needed

# # Loop over each (initial_time, end_time) pair
# for pair in "${time_pairs[@]}"; do
#     # Read initial_time and end_time from the pair
#     read initial_time end_time <<< "$pair"

#     cadence=100000

#     # Reset start_time to initial_time at the beginning of each pair's loop
#     start_time=$initial_time

#     while [ "$(($start_time + $duration))" -le "$end_time" ]; do
#         noise_frame="/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/4096/layer0/Hrec-HOFT-1265127585-100000/original-1265127585-4096.h5"
#         inj_frame="/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Injections/new_injection/15-415_Hz/injSignal/V1_BBH_inj_${start_time}_4096.hdf5"
#         output_frame="/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Injections/new_injection/15-415_Hz/noisePlusSignal"

#         # Echo command for verification or logging
#         echo "Processing $noise_frame"

#         # Run the Python script with constructed arguments
#         python add_injection_strain_to_noise_strain.py --strain-channel "V1:Hrec_hoft_raw_20000Hz" --chanslist "/home/weizmann.kiendrebeogo/DeepClean/DeepClean_CIT/Multi-train-process/First-run/witnesses/witnesses_142-162_Hz.ini" --noise-frame "$noise_frame" --inj-frame "$inj_frame" --output-frame "$output_frame"

#         # Update start_time for the next iteration
#         start_time=$(($start_time + $duration))
#     done
# done
# echo "All injections added to noise strain."
