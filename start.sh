#!/bin/bash

# Start FastAPI in the background
echo "Starting FastAPI backend..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready (optional but good practice)
sleep 5

# Start Streamlit in the foreground (Hugging Face expects port 7860)
echo "Starting Streamlit dashboard..."
streamlit run src/dashboard/app.py --server.port 7860 --server.address 0.0.0.0
