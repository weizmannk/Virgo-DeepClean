# Script to write the first two channels into 'layer0.ini',
# and then write each subsequent channel into its own file: 'layer1.ini', 'layer2.ini', etc.

input_filename = './witnesses-sequential/witnesses_142-162_all.ini'  # Path to the input file


# Read all non-empty lines from the input file
with open(input_filename, 'r') as infile:
    lines = [l.strip() for l in infile if l.strip()]

# Write the first two channels into 'layer0.ini'
with open('./witnesses-sequential/layer0.ini', 'w') as outfile:
    outfile.write(f"{lines[0]}\n")
    outfile.write(f"{lines[1]}")

# Write each subsequent channel into its own file: 'layer1.ini', 'layer2.ini', ...
for idx, channel in enumerate(lines[2:], start=1):
    filename = f"./witnesses-sequential/layer{idx}.ini"
    with open(filename, 'w') as outfile:
        outfile.write(f"layer{idx-1}\n")
        outfile.write(f"{channel}")
