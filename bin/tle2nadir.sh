#!/bin/bash

# Define virtual environment directory
VENV_DIR="venv"

# Check if venv exists, if not, create it
if [ ! -d "$VENV_DIR" ]; then
    echo "Initializing virtual environment..."
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source $VENV_DIR/bin/activate
fi

# Run the Python script
python tle2nadir_gui.py "$@"

# Deactivate (optional)
deactivate