#!/bin/bash

# Define the directory where to move incorrect datasets
target_dir="./wrong_dataset"

# Create the target directory if it doesn't exist
mkdir -p "$target_dir"

# Iterate through all directories in the current folder
for dir in TNL2K_test/*/; do
    # Check if the directory contains language.txt
    if [ ! -f "$dir/language.txt" ]; then
        echo "Moving $dir to $target_dir"
        mv "$dir" "$target_dir"
    fi
done