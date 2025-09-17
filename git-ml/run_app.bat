@echo off
echo Starting GitHub Security Anomaly Detection Web Interface...
echo.
echo The application will be available at: http://localhost:8503
echo Press Ctrl+C to stop the application
echo.

REM Activate virtual environment and run Streamlit app
C:\Users\User\Desktop\Projects\git-ml\.venv\Scripts\streamlit.exe run ui\app.py --server.port 8503

echo.
echo Application stopped.
pause
