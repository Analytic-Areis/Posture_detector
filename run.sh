#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the script directory
cd "$DIR"

# Activate the virtual environment
source venv/bin/activate

# Run the python script
python main.py
