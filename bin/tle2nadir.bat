@echo off
setlocal

:: Define the virtual environment directory
set VENV_DIR=venv

:: Check if venv exists, if not, create it
if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

:: Activate the virtual environment
call %VENV_DIR%\Scripts\activate

:: Install dependencies if needed
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

:: Run the Python script
python tle2nadir_gui.py %*

:: Deactivate (optional)
deactivate

endlocal