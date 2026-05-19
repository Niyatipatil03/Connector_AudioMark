@echo off
title AudioMark Annotator — Mahindra
echo.
echo  ================================================
echo   AudioMark Annotator v1.0
echo   Mahindra Connector Lock Detection
echo  ================================================
echo.
python --version >nul 2>&1
if %errorlevel% neq 0 (echo [ERROR] Python not found. Install from python.org & pause & exit /b 1)
echo  Installing packages...
pip install fastapi uvicorn python-multipart librosa soundfile scipy noisereduce --quiet
echo.
echo  Starting server...
echo.
echo  Open browser:  http://localhost:8100
echo  Press Ctrl+C to stop
echo.
python server.py
pause
