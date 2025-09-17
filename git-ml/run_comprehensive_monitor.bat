@echo off
echo =============================================
echo   Comprehensive Repository Security Monitor
echo =============================================
echo.
echo Starting comprehensive monitoring dashboard...
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install requirements if needed
if not exist ".venv\pyvenv.cfg" (
    echo Installing requirements...
    pip install -r requirements.txt
)

echo.
echo ================================================
echo  🔒 COMPREHENSIVE REPOSITORY SECURITY MONITOR
echo ================================================
echo.
echo Opening comprehensive dashboard at:
echo http://localhost:8507
echo.
echo Features:
echo - Complete repository information dashboard
echo - Real-time security monitoring
echo - Detailed metrics and analytics
echo - Security risk assessments
echo - Comprehensive alert management
echo.
echo Press Ctrl+C to stop monitoring
echo.

REM Start the comprehensive monitoring app
streamlit run ui/comprehensive_realtime_app.py --server.port=8507 --server.headless=true

pause
