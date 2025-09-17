@echo off
echo Starting Git Repository Security Analyzer...
echo.

REM Activate virtual environment and run Git app
C:\Users\User\Desktop\Projects\git-ml\.venv\Scripts\streamlit.exe run ui/git_app.py --server.port 8503

echo.
echo Application stopped. Press any key to continue...
pause > nul
