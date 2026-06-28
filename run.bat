@echo off
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist .venv (
    echo [Gesture Meme] Virtual environment not found. Setting up compatible Python environment...
    
    :: Try using the py launcher with Python 3.11
    py -3.11 -m venv .venv >nul 2>&1
    if errorlevel 1 (
        :: Fallback to default python if py -3.11 is not available
        echo [Gesture Meme] Python 3.11 launcher not found, falling back to system default 'python'...
        python -m venv .venv >nul 2>&1
    )
    
    if not exist .venv (
        echo.
        echo [ERROR] Could not create virtual environment. 
        echo Please ensure Python is installed and added to your system PATH.
        echo Recommended version: Python 3.11
        echo.
        pause
        exit /b 1
    )
    
    echo [Gesture Meme] Installing dependencies... This may take a minute...
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Dependency installation failed.
        echo.
        pause
        exit /b 1
    )
    echo [Gesture Meme] Setup complete!
    echo.
)

echo Starting Gesture Meme application...
.venv\Scripts\python.exe main.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed or failed to start.
    echo.
    pause
)
