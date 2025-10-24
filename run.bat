@echo off
setlocal EnableDelayedExpansion
echo Starting AI Image Generator...
echo.

REM Use the copied InvokeAI virtual environment
set VENV_FOUND=0

if exist "venv\Scripts\activate.bat" (
    set VENV_PATH=%CD%\venv\Scripts
    set VENV_FOUND=1
) else (
    echo ❌ Local venv not found! Please create a virtual environment first.
    echo.
    pause
    exit /b 1
)

if !VENV_FOUND!==0 (
    echo ❌ No virtual environment found!
    echo.
    echo Please create a virtual environment and install requirements.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment at: !VENV_PATH!
call "!VENV_PATH!\activate.bat"

REM Verify Python is available in venv
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not available in the virtual environment.
    echo Please check your virtual environment setup.
    pause
    exit /b 1
)

echo ✅ Virtual environment activated successfully.
echo Python version: & python --version
echo Python executable: & where python
echo.

REM Test diffusers import before running main app
echo Testing diffusers import...
python -c "import diffusers; print('✅ diffusers import successful')" 2>nul
if errorlevel 1 (
    echo ❌ diffusers import failed! Installing...
    pip install diffusers==0.32.1
    if errorlevel 1 (
        echo ❌ Failed to install diffusers
        pause
        exit /b 1
    )
    echo ✅ diffusers installed successfully
)
echo.

REM Run the application using venv Python directly
"%~dp0venv\Scripts\python.exe" main.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ❌ Application exited with error code !errorlevel!
    echo.
    echo If you see import errors, run: pip install -r requirements.txt
    pause
)
endlocal
