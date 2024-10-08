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

# Loop through each .tar.gz file in the source directory
for tar_file in "$SOURCE_DIR"/*.tar.gz; do
    # Extract the .tar.gz file to a temporary directory
    temp_dir=$(mktemp -d) # Create a temporary directory
    tar -xzvf "$tar_file" -C "$temp_dir" # Extract the tar.gz file contents
    
    # Move all extracted directories/files from the temporary directory to the target directory
    mv "$temp_dir"/*/* "$TARGET_DIR"
    
    # Remove the temporary directory
    rm -rf "$temp_dir"
done