import pandas as pd
from scipy import stats
import os

# Create the output directory if it doesn't exist
# fmin, fmax = 98, 110
outdir = "data"
os.makedirs(outdir, exist_ok=True)
directory = "data"


for freq in ["98-110_Hz", "197-208_Hz", "142-162_Hz", "15-415_Hz"]:

    print(f"\n {freq}\n")

    # Load the data into a DataFrame
    data = pd.read_csv(os.path.join(directory, f"BNS_inspiral_range_{freq}.csv"))

    # Calculate the difference in inspiral range (DC - ORG)
    data["Inspiral Range Difference"] = (
        data["BNS Inspiral Range DC"] - data["BNS Inspiral Range ORG"]
    )

    # Perform a paired t-test
    t_stat, p_value = stats.ttest_rel(
        data["BNS Inspiral Range ORG"], data["BNS Inspiral Range DC"]
    )

    print(f"T-statistic: {t_stat}, P-value: {p_value}\n")

    # Check for statistical significance
    if p_value < 0.05:
        print(
            "There is a significant difference between the original and DeepCleaned data."
        )
    else:
        print(
            "There is no significant difference between the original and DeepCleaned data.\n"
        )

    # Calculate the mean improvement
    mean_improvement = data["Inspiral Range Difference"].mean()
    print(
        f"Mean improvement in BNS Inspiral Range after DeepClean: {mean_improvement}\n"
    )

    # # Output the results to a new CSV file
    # data.to_csv(os.path.join(outdir, f'BNS_Inspiral_Range_Comparison_{freq}_32s.csv'), index=False)

    # Assuming 'mean_original_range' is the average of the BNS Inspiral Range ORG column
    mean_original_range = data["BNS Inspiral Range ORG"].mean()

    # Calculate the percentage improvement
    percentage_improvement = (mean_improvement / mean_original_range) * 100

    print(
        f"\n Percentage improvement in BNS Inspiral Range after DeepClean: {percentage_improvement:.3f}%\n"
    )
