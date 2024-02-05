import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


fmin, fmax = 145, 155

# Define the output directory for plots and ensure it exists
outdir = "output"
os.makedirs(outdir, exist_ok=True)

# Load the pre-cleaning and post-cleaning SNR data
snr_pre_clean = pd.read_csv('./output/Pre-clean_snr_peaks_all_classifications_145-155.csv')
snr_post_clean = pd.read_csv('./output/Post-clean_snr_peaks_all_classifications_145-155.csv')

# Merge the dataframes on the 'Index' column to compare pre and post SNR values
merged_data = pd.merge(snr_pre_clean, snr_post_clean, on='Index', suffixes=('_pre', '_post'))

# Calculate SNR improvement percentage
merged_data['SNR_Improvement_Percentage'] = 100 * (merged_data['Peak_SNR_post'] - merged_data['Peak_SNR_pre']) / merged_data['Peak_SNR_pre']

# Filter the merged data for different classifications
isco_events = merged_data[merged_data['Classification_pre'] == 'isco']
merger_events = merged_data[merged_data['Classification_pre'] == 'merger']
below_fmin_events = merged_data[merged_data['Classification_pre'] == 'below_fmin']
above_fmax_events = merged_data[merged_data['Classification_pre'] == 'above_fmax']

# Plotting SNR comparison and improvement
plt.figure(figsize=(14, 7))

# SNR comparison plot
plt.subplot(1, 2, 1)
if not isco_events.empty:
    plt.scatter(isco_events['Peak_SNR_Time_pre'], isco_events['Peak_SNR_pre'], label='Pre-Clean ISCO', color='blue', marker='o')
    plt.scatter(isco_events['Peak_SNR_Time_post'], isco_events['Peak_SNR_post'], label='Post-Clean ISCO', color='red', marker='x')
if not merger_events.empty:
    plt.scatter(merger_events['Peak_SNR_Time_pre'], merger_events['Peak_SNR_pre'], label='Pre-Clean Merger', color='green', marker='o')
    plt.scatter(merger_events['Peak_SNR_Time_post'], merger_events['Peak_SNR_post'], label='Post-Clean Merger', color='orange', marker='x')
if not below_fmin_events.empty:
    plt.scatter(below_fmin_events['Peak_SNR_Time_pre'], below_fmin_events['Peak_SNR_pre'], label='Pre-Clean Below fmin', color='purple', marker='o')
    plt.scatter(below_fmin_events['Peak_SNR_Time_post'], below_fmin_events['Peak_SNR_post'], label='Post-Clean Below fmin', color='pink', marker='x')
if not above_fmax_events.empty:
    plt.scatter(above_fmax_events['Peak_SNR_Time_pre'], above_fmax_events['Peak_SNR_pre'], label='Pre-Clean Above fmax', color='lightblue', marker='o')
    plt.scatter(above_fmax_events['Peak_SNR_Time_post'], above_fmax_events['Peak_SNR_post'], label='Post-Clean Above fmax', color='black', marker='x')
plt.xlabel('Time (s)')
plt.ylabel('Peak SNR')
plt.title('SNR Comparison Over Time for All Classifications')
plt.legend()

# SNR improvement percentage plot

# Customize the bar width
bar_width = 0.3

plt.subplot(1, 2, 2)
if not isco_events.empty:
    sns.barplot(x='Index', y='SNR_Improvement_Percentage', data=isco_events, color='skyblue', label='ISCO',  width=bar_width)
if not merger_events.empty:
    sns.barplot(x='Index', y='SNR_Improvement_Percentage', data=merger_events, color='lightgreen', label='Merger',  width=bar_width)
if not below_fmin_events.empty:
    sns.barplot(x='Index', y='SNR_Improvement_Percentage', data=below_fmin_events, color='purple', label='Below fmin',  width=bar_width)
if not above_fmax_events.empty:
    sns.barplot(x='Index', y='SNR_Improvement_Percentage', data=above_fmax_events, color='pink', label='Above fmax',  width=bar_width)
    


plt.xlabel('Injection Index')
plt.ylabel('SNR Improvement Percentage (%)')
plt.title('SNR Improvement by DeepClean')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(outdir, f'SNR_Improvement_Analysis_{fmin}-{fmax}.png'))
plt.show()

# Statistical Summary
mean_improvement_isco = isco_events['SNR_Improvement_Percentage'].mean() if not isco_events.empty else 0
mean_improvement_merger = merger_events['SNR_Improvement_Percentage'].mean() if not merger_events.empty else 0
mean_improvement_below_fmin = below_fmin_events['SNR_Improvement_Percentage'].mean() if not below_fmin_events.empty else 0
mean_improvement_above_fmax = above_fmax_events['SNR_Improvement_Percentage'].mean() if not above_fmax_events.empty else 0


print(f"Mean SNR Improvement for ISCO: {mean_improvement_isco:.2f}%")
print(f"Mean SNR Improvement for Merger: {mean_improvement_merger:.2f}%")
print(f"Mean SNR Improvement for Below fmin: {mean_improvement_below_fmin:.2f}%")
print(f"Mean SNR Improvement for Above fmax: {mean_improvement_above_fmax:.2f}%")
