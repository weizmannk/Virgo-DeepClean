#!/bin/bash


# s1=1265127585; t1=1265172641
# s2=1265176737; t2=1265225889
# s3=1265227585; t3=1265272641
# s4=1265276737; t4=1265325889
# s5=1265327585; t5=1265352161

# # Duration
# duration=4096

# # # Define an array of pairs
# time_pairs=("$s1 $t1" "$s2 $t2" "$s3 $t3" "$s4 $t4" "$s5 $t5")

#Define start and end times for each pair
s1=1265127585; t1=1265225889
s2=1265227585; t2=1265325889
s3=1265327585; t3=1265352161

# Duration
duration=4096


# Define an array of pairs
time_pairs=("$s1 $t1" "$s2 $t2" "$s3 $t3")

# time_pairs=("$s1 $t1" "$s2 $t2")

# Loop over each (initial_time, end_time) pair
for pair in "${time_pairs[@]}"; do
    # Read initial_time and end_time from the pair
    read initial_time end_time <<< "$pair"


    # Reset start_time to initial_time at the beginning of each pair's loop
    start_time=$initial_time

    while [ "$(($start_time ))" -le "$end_time" ]; do

        echo "Creating injection file for start time: $start_time"
    python create_injection_strain.py --interferometers V1 --inj-gap 32 --prior-fil ./priors/BBH_V1_197-215_Hz.prior --start-time $start_time --duration $duration --frame-duration $duration --sampling-frequency 4096 --inject --outdir ./Rework/197-208_Hz/injSignal

        # Update start_time for the next iteration
        start_time=$(($start_time + $duration))
    done
done

echo "All injection files created."
