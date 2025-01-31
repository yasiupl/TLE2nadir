@echo off
setlocal

:: Define the virtual environment directory
set VENV_DIR=venv

:: Check if venv exists, if not, create it
if not exist %VENV_DIR% (
    echo Initializing virtual environment...
    python -m venv %VENV_DIR%
    call %VENV_DIR%\Scripts\activate
    echo Installing dependencies...
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo Virtual environment found. Activating...
    call %VENV_DIR%\Scripts\activate
)

:: Run the Python script
python tle2nadir_gui.py %*

:: Deactivate (optional)
deactivate

endlocal