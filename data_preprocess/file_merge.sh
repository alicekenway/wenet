#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_file>"
    exit 1
fi

input_directory="$1"
output_file="$2"

# Check if the input directory exists
if [ ! -d "$input_directory" ]; then
    echo "Error: Directory '$input_directory' does not exist."
    exit 1
fi

# Create or empty the output file
> "$output_file"

# Loop through all files in the input directory
for file in "$input_directory"/*; do
    # Check if it's a regular file
    if [ -f "$file" ]; then
        cat "$file" >> "$output_file"
        echo -e "\n" >> "$output_file" # Add a newline after each file
    fi
done

echo "All files from '$input_directory' have been merged into '$output_file'."
