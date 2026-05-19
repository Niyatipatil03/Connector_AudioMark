#!/bin/bash
echo ""
echo "AudioMark Annotator v1.0 — Mahindra"
echo ""
echo "Installing packages..."
pip install fastapi uvicorn python-multipart librosa soundfile scipy noisereduce --quiet
echo ""
echo "Starting server..."
echo "Open browser: http://localhost:8100"
echo ""
python3 server.py
