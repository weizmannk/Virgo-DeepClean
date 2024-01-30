import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gwpy.timeseries as ts
import os

datapath = "./" + os.path.join("injection_Dir_98_108_Hz")
os.makedirs(datapath, exist_ok=True)

# Load the strain data
file_path = "./injection_Dir_98_108_Hz/V1_BBH_inj_1265127585.0_1265128609.0.hdf5"
V1_inj = ts.TimeSeries.read(file_path)

print(V1_inj.times[0])
print(V1_inj)


# Load the injections data
injections = pd.read_csv('./injection_Dir_98_108_Hz/INJ-1265127585-1024.csv')
injections = injections[0:3]
inj_gap = 32
# Set up the plot for 9 subplots
plt.figure(figsize=(15, 12))


# Determine the number of subplots for visualization
num_injections = len(injections)
num_rows = int(num_injections**0.5) + 1
num_cols = (num_injections // num_rows) + 1
plt.figure(figsize=(15, num_rows * 3))
   
for index, row in injections.iterrows():
    start_time = row['geocent_time'] 
    end_time = row['geocent_time'] + inj_gap
    start_index = int((start_time - V1_inj.t0.value) * V1_inj.sample_rate.value)
    
    print(start_index)
    end_index = int((end_time - V1_inj.t0.value) * V1_inj.sample_rate.value)

    trigtime = row['geocent_time']
    
    signal = V1_inj[start_index:end_index]
    
    plt.subplot(num_rows, num_cols, index + 1)
    plt.plot(np.array(signal.times), signal.value, label='V1 Injected Signal')
    
    # Assuming wf_pols and wfg are defined and represent the waveform polarizations and generator
    # Uncomment and modify the following line as per your waveform data
    # plt.plot(wfg - wfg[0] + trigtime - 3.8, wf_pols['plus'], ls='dashed', label='Injected Waveform')
    
    plt.xlim(trigtime -1.2, trigtime+0.5)
    plt.ylim(-5e-21/2, 5e-21/2)
    plt.axvline(trigtime, color='red', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.title(f'Injection {index}')
    plt.legend()

# Adjust the layout
plt.tight_layout()

# Save the plot
plot_filename = os.path.join(datapath, 'gw_signal_injections.pdf')
plt.savefig(plot_filename, format='pdf')
print(f"Plot saved as {plot_filename}")

# Optionally, show the plot
# plt.show()

