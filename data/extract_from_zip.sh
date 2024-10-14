#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/target_directory"
    exit 1
fi

# Set the target directory from the argument
TARGET_DIR=$1

# Define the source directory where your tar.gz files are located
SOURCE_DIR="../data_zip/TNL2K_test"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Initialize a variable to keep track of the number of extracted directories/files
num_extracted=0

# Loop through each .tar.gz file in the source directory
for tar_file in "$SOURCE_DIR"/*7.tar.gz; do
    
    tar -xzvf "$tar_file" -C "$TARGET_DIR" # Extract the tar.gz file contents
    
    # Calculate the number of extracted directories/files
    extracted_count=$(ls -1 "$TARGET_DIR" | wc -l)
    echo "Extracted $extracted_count directories/files from $tar_file to $TARGET_DIR"
    num_extracted=$((num_extracted + extracted_count))
done

# Print the number of extracted directories/files
echo "Extracted $num_extracted directories/files to $TARGET_DIR"