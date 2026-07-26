#!/mnt/users/jinyang_wang/miniconda3/miniconda3/bin/python
import sys
import random

def filter_file(input_file, output_file, threshold):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if random.random() < threshold:
                outfile.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <threshold> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    threshold = float(sys.argv[2])
    output_file = sys.argv[3]

    filter_file(input_file, output_file, threshold)
    print(f"Filtering complete. Output written to {output_file}")
