import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Define the output directory for plots and ensure it exists
outdir = "output"
os.makedirs(outdir, exist_ok=True)

# Load the SNR data from pre and post cleaning
snr_pre_clean = pd.read_csv('./output/snr_values_pre_clean.csv')
snr_post_clean = pd.read_csv('./output/snr_values_post_clean.csv')

# Merge the datasets for plotting
snr_data_merged = pd.merge(snr_pre_clean, snr_post_clean, on='Injection-num', suffixes=('_pre', '_post'))

# Plotting
plt.figure(figsize=(14, 6))

# Left panel - Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(snr_data_merged['Time_pre'], snr_data_merged['SNR_pre'], alpha=0.5, label='Pre-Clean')
plt.scatter(snr_data_merged['Time_post'], snr_data_merged['SNR_post'], alpha=0.5, label='Post-Clean')
plt.xlabel('Time (s)')
plt.ylabel('SNR')
plt.title('Scatter Plot of SNR Before and After Cleaning')
plt.legend()

# Right panel - Cumulative plot
plt.subplot(1, 2, 2)
sns.ecdfplot(data=snr_data_merged, x='SNR_pre', stat="count", complementary=True, label='Pre-Clean')
sns.ecdfplot(data=snr_data_merged, x='SNR_post', stat="count", complementary=True, label='Post-Clean')
plt.xlabel('SNR')
plt.ylabel('Complementary Cumulative Count')
plt.title('Cumulative SNR Improvement')
plt.legend()

plt.suptitle('SNR Analysis Before and After Cleaning')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'SNR_Analysis_Comparison.png'))
plt.show()



# Compute SNR improvement
# Calculate SNR improvement percentage
snr_improvement = 100 * (snr_post_clean['SNR'] - snr_pre_clean['SNR']) / snr_pre_clean['SNR']

# Calculate statistical metrics
mean_improvement = snr_improvement.mean()
median_improvement = snr_improvement.median()
std_dev_improvement = snr_improvement.std()

print(f"Mean SNR Improvement: {mean_improvement:.2f}%")
print(f"Median SNR Improvement: {median_improvement:.2f}%")
print(f"Standard Deviation of SNR Improvement: {std_dev_improvement:.2f}%")
