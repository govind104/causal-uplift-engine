import subprocess
import time
import sys
import os
import requests

print("=" * 60)
print("CAUSAL UPLIFT ENGINE - STARTING SERVICES")
print("=" * 60)

# 1. Start FastAPI Backend
print("[1/2] Starting FastAPI backend on port 8000...")
backend_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=sys.stdout,
    stderr=sys.stderr
)

# 2. Wait for Backend to be Ready
print("[2/2] Waiting for backend health check...")
ready = False
for i in range(30):  # Wait up to 30 seconds
    try:
        # Check root endpoint or docs
        resp = requests.get("http://localhost:8000/", timeout=1)
        if resp.status_code == 200 or resp.status_code == 404: # 404 is fine (FastAPI running)
            print(f"      Backend ready after {i+1} seconds!")
            ready = True
            break
    except requests.exceptions.ConnectionError:
        pass
    time.sleep(1)
    if i % 5 == 0 and i > 0:
        print(f"      Still waiting... ({i}s)")

if not ready:
    print("      WARNING: Backend did not respond in 30s. Starting Dashboard anyway...")

print("=" * 60)
print("      Starting Streamlit on port 7860...")
print("=" * 60)

# 3. Start Streamlit Frontend
# Note: usage of sys.executable ensures we use the same venv python
subprocess.run([
    sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py",
    "--server.port", "7860",
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false"
])
