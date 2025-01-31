#!/bin/bash

# Define virtual environment directory
VENV_DIR="venv"

# Check if venv exists, if not, create it
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Install dependencies if needed
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run the Python script
python tle2nadir_gui.py "$@"

# Deactivate (optional)
deactivate