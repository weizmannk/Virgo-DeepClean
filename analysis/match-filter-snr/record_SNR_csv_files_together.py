import os
import pandas as pd
from glob import glob

# Define the output directory for plots and ensure it exists
outdir = "SNR-Peak"
os.makedirs(outdir, exist_ok=True)

# Directory containing your CSV files


for filedir in ["98-110_Hz", "142-162_Hz", "197-208_Hz", "15-415_Hz"]:
    directory = f"./{filedir}/matchedfilter_SNR_Freq/4096_32s"

    for step in ["Pre", "Post"]:

        # Pattern to match the CSV files starting with 'Pre' or 'Post'
        pattern = os.path.join(directory, f"{step}*.csv")  # Use an f-string here

        # Find all matching files
        csv_files = glob(pattern)

        # List to hold data from each CSV file
        dataframes = []

        for filename in csv_files:
            # Read the current CSV file into a DataFrame
            df = pd.read_csv(filename)

            df.drop(columns=["Index"], inplace=True)

            # Append the DataFrame to the list
            dataframes.append(df)

        # Concatenate all DataFrames in the list into a single DataFrame
        concatenated_df = pd.concat(dataframes, ignore_index=True)

        # Sort the DataFrame by 'Peak_SNR_Time' column
        sorted_df = concatenated_df.sort_values(by="Peak_SNR_Time", ascending=True)

        # Reset the index, drop the old one, and adjust to start from 1
        sorted_df.reset_index(drop=True, inplace=True)
        sorted_df.index += 1

        # Define the path for the output CSV file
        output_file = os.path.join(outdir, f"{step}_SNR_all_{filedir}_32s.csv")

        # Save the sorted DataFrame to a new CSV file, including the new index
        sorted_df.to_csv(output_file, index=True, index_label="Index")

        print(
            f'All files starting with "{step}" have been combined, sorted, and indexed into {output_file}'
        )
