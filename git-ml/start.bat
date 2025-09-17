@echo off
title GitHub Security Anomaly Detection Suite
echo ===============================================================================
echo 🔐 GITHUB SECURITY ANOMALY DETECTION SUITE
echo ===============================================================================
echo.
echo Choose an option:
echo.
echo [1] 🌐 Web Interface (Streamlit App)
echo [2] 💻 Command Line Demo  
echo [3] 📊 Git Repository Analysis
echo [4] 🚀 Main Menu (All Options)
echo [5] ⚡ Real-Time Monitor (Enhanced)
echo [0] ❌ Exit
echo.
set /p choice="Enter your choice (0-5): "

if "%choice%"=="1" (
    echo Starting Web Interface...
    call run_app.bat
) else if "%choice%"=="2" (
    echo Starting Command Line Demo...
    echo This may take a moment to generate data and train the model...
    .venv\Scripts\python.exe demo.py
    pause
) else if "%choice%"=="3" (
    echo Starting Git Analysis...
    call run_git_app.bat
) else if "%choice%"=="4" (
    echo Starting Main Menu...
    call run_main.bat
) else if "%choice%"=="5" (
    echo Starting Enhanced Real-Time Monitor...
    call run_realtime_improved.bat
) else if "%choice%"=="0" (
    echo Goodbye!
    exit
) else (
    echo Invalid choice. Please try again.
    pause
    goto start
)
