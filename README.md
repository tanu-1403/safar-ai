# 🛣️ Safar AI
## *Har Safar, Surakshit Safar*
### हर सफर, सुरक्षित सफर — Every Journey, A Safe Journey

> **AI-Powered Spectral Reflectivity Intelligence for National Highways**
> Built for NHAI Hackathon 2024 | Digital Twin · Hyperspectral Vision · Edge AI · Degradation Forecasting

---

## Table of Contents

1. [What is Safar AI?](#1-what-is-safar-ai)
2. [The Problem](#2-the-problem)
3. [Our Solution](#3-our-solution)
4. [System Architecture](#4-system-architecture)
5. [Module Reference](#5-module-reference)
6. [Project Structure](#6-project-structure)
7. [Prerequisites](#7-prerequisites)
8. [Step-by-Step Deployment Guide](#8-step-by-step-deployment-guide)
9. [Running the Application](#9-running-the-application)
10. [Dashboard Guide](#10-dashboard-guide)
11. [Configuration Reference](#11-configuration-reference)
12. [API and Data Formats](#12-api-and-data-formats)
13. [Troubleshooting](#13-troubleshooting)
14. [Scaling to National Deployment](#14-scaling-to-national-deployment)
15. [Tech Stack](#15-tech-stack)
16. [License](#16-license)

---

## 1. What is Safar AI?

**Safar AI** (सफर AI) is a production-grade intelligent highway monitoring system that combines:

- **Hyperspectral imaging simulation** (31 bands, 400–700 nm) to measure road surface reflectivity beyond what RGB cameras can see
- **Digital Twin architecture** — every highway segment has a living virtual counterpart storing its full history
- **AI degradation forecasting** (ARIMA + LSTM) to predict when road markings will fail before they become dangerous
- **Edge deployment** — runs on a Raspberry Pi 4 or Jetson Nano inside a patrol vehicle
- **Explainable AI** — field engineers understand exactly why the model flagged a segment

The name *Safar* (सफर) means **journey** in Hindi and Urdu.
The tagline *Har Safar, Surakshit Safar* means **Every Journey, A Safe Journey**.

---

## 2. The Problem

| Metric | India Today |
|--------|------------|
| National highway network | 1,46,000+ km |
| Faded-marking accidents/year | ~15,000 (estimated) |
| Current inspection coverage | ~5% of network annually |
| Maintenance model | Reactive — after accident or complaint |
| Digital reflectivity records | None |

Road markings and retroreflective surfaces degrade due to traffic wear, monsoon rain, UV aging, and contamination. Current practice is manual eyeball inspection with paper reports. There is no predictive model, no time-series data, and no automated alert system.

---

## 3. Our Solution

```
Camera Frame  ──┐
                ├──► Spectral Fusion ──► Reflectivity Score [0–1]
Hyperspectral  ──┘         │
  Simulation               ▼
                     GBM Regressor ◄── 15 engineered features
                           │
                           ▼
                    Digital Twin Registry
                    (per-segment history)
                           │
                           ▼
                   ARIMA / LSTM Forecast ──► 60-day prediction + CI
                           │
                           ▼
              ┌────────────┴────────────┐
              ▼                         ▼
       Maintenance Alert         Streamlit Dashboard
       (WhatsApp / SMS)          (Map · Charts · Explainability)
```

---

## 4. System Architecture

| Layer | Components | Technology |
|-------|-----------|-----------|
| Input | RGB camera, spectral simulator | OpenCV, NumPy, SciPy |
| Processing | CNN extractor, GBM regressor | TensorFlow/Keras, Scikit-learn |
| Intelligence | Digital twin, ARIMA/LSTM | statsmodels, pandas |
| Edge | INT8 model, telemetry, delta encoder | NumPy-only, JSON |
| Explainability | PDP, what-if, counterfactual, SHAP | SHAP, custom |
| Output | Dashboard, alerts, audit report | Streamlit, Plotly |

---

## 5. Module Reference

| File | Purpose | Lines |
|------|---------|-------|
| `utils.py` | Constants, thresholds, helpers | 120 |
| `modules/ingestion.py` | OpenCV pipeline, visual feature extraction | 293 |
| `modules/spectral.py` | 31-band hyperspectral engine, noise models | 375 |
| `modules/model.py` | CNN extractor + GBM regressor + SHAP | 351 |
| `modules/digital_twin.py` | HighwaySegment twin, registry, persistence | 376 |
| `modules/prediction.py` | ARIMA/LSTM forecaster, maintenance scheduler | 512 |
| `modules/edge_deployment.py` | INT8 model, telemetry, delta encoder | 290 |
| `modules/explainability.py` | PDP, what-if, counterfactual, audit report | 340 |
| `app.py` | 6-tab Streamlit dashboard | 970 |
| `generate_dataset.py` | One-shot data + model generator | 113 |
| `run_demo.py` | Headless full-system verification | 200 |

**Total: ~3,940 lines of production-grade Python**

---

## 6. Project Structure

```
safar_ai/
│
├── app.py                         # Streamlit dashboard (main entry point)
├── generate_dataset.py            # Run FIRST — generates all data + model
├── run_demo.py                    # Headless system verification
├── utils.py                       # Shared constants and helpers
├── requirements.txt
├── README.md
│
├── modules/
│   ├── ingestion.py
│   ├── spectral.py
│   ├── model.py
│   ├── digital_twin.py
│   ├── prediction.py
│   ├── edge_deployment.py
│   └── explainability.py
│
├── data/                          # Auto-generated on first run
│   ├── synthetic_spectral_data.csv
│   ├── training_features.csv
│   ├── segment_summary.csv
│   ├── active_alerts.csv
│   ├── maintenance_recommendations.csv
│   ├── digital_twin_state.json
│   ├── pdp_features.json
│   ├── edge_telemetry_demo.json
│   └── demo_audit_report.json
│
└── models/
    └── reflectivity_predictor.joblib
```

---

## 7. Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04 / Windows 10 / macOS 12 | Ubuntu 22.04 LTS |
| Python | 3.9 | 3.11 |
| RAM | 4 GB | 8 GB |
| Storage | 2 GB free | 5 GB free |
| CPU | 4-core | 8-core |

### Check your Python version

```bash
python3 --version    # must be 3.9 or higher
pip3 --version
```

If Python 3.9+ is not installed:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install python3.11 python3.11-pip python3.11-venv -y

# macOS (Homebrew)
brew install python@3.11

# Windows — download from https://python.org/downloads
# During install: check "Add Python to PATH"
```

---

## 8. Step-by-Step Deployment Guide

---

### A. Local Development (Laptop / Desktop)

This is the fastest way to get Safar AI running.

#### Step 1 — Get the code

```bash
# If you have the ZIP file
unzip nhai_digital_twin_FINAL.zip
mv nhai_digital_twin safar_ai
cd safar_ai
```

#### Step 2 — Create a virtual environment

```bash
python3 -m venv venv

# Activate — Linux / macOS
source venv/bin/activate

# Activate — Windows Command Prompt
venv\Scripts\activate.bat

# Activate — Windows PowerShell
venv\Scripts\Activate.ps1

# You should see (venv) in your terminal prompt now
```

#### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs approximately 800 MB of packages. It takes 3–10 minutes depending on your internet speed.

> **Faster install — skip TensorFlow:**
> If you do not need the LSTM model or CNN extractor, you can install only the core packages:
> ```bash
> pip install numpy pandas scipy opencv-python-headless scikit-learn \
>             streamlit plotly statsmodels shap joblib Pillow tqdm
> ```
> All dashboard features remain fully functional. The system auto-detects TF is unavailable and uses the GBM + ARIMA path only.

#### Step 4 — Generate data and train the model

```bash
python generate_dataset.py
```

Expected output (takes 8–15 seconds):

```
════════════════════════════════════════════════════════════
  Safar AI — Dataset & Model Generator
  Har Safar, Surakshit Safar
════════════════════════════════════════════════════════════

[1/5] Generating synthetic hyperspectral dataset...
   ✅ Dataset: 3,600 rows × 10 columns

[3/5] Training Gradient Boosting Reflectivity Predictor...
   ✅ Test MAE: 0.0063  |  Test R²: 0.9981

[4/5] Constructing Digital Twin registry...
   ✅ Digital Twin: 20 segments created

[5/5] Fitting ARIMA forecasters for all segments...
   ✅ Forecasts computed for 20 segments

  Safar AI is ready.
  Tagline: Har Safar, Surakshit Safar
  Next step → streamlit run app.py
```

#### Step 5 — (Optional) Run system verification

```bash
python run_demo.py
```

This exercises every module end-to-end and prints a full report. Useful to confirm everything works before the dashboard.

#### Step 6 — Launch the dashboard

```bash
streamlit run app.py
```

Streamlit will print:

```
  You can now view your Streamlit app in your browser.
  Local URL:   http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open `http://localhost:8501` in your browser.

---

### B. Docker Deployment

Use Docker for a clean, reproducible environment — ideal for team sharing and demo machines.

#### Step 1 — Create Dockerfile

Save this as `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python generate_dataset.py

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app.py"]
```

#### Step 2 — Create .dockerignore

```
venv/
__pycache__/
*.pyc
.git/
```

#### Step 3 — Build and run

```bash
# Build (takes 5–10 minutes first time)
docker build -t safar-ai:latest .

# Run
docker run -p 8501:8501 safar-ai:latest

# Run with persistent data (saves twin state across restarts)
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  safar-ai:latest
```

Open `http://localhost:8501`

#### Step 4 — Docker Compose (optional)

Save as `docker-compose.yml`:

```yaml
version: '3.8'
services:
  safar-ai:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
```

```bash
docker-compose up -d        # start in background
docker-compose logs -f      # view logs
docker-compose down         # stop
```

---

### C. Cloud Deployment

#### Option 1 — AWS EC2

```bash
# 1. Launch EC2 instance
#    Type: t3.medium (2 vCPU, 4 GB RAM)  ~$0.04/hr
#    AMI:  Ubuntu Server 22.04 LTS
#    Security group: open port 8501 (custom TCP) + 22 (SSH)

# 2. SSH in
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# 3. Install system packages
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv git -y

# 4. Upload or clone your project
scp -r safar_ai/ ubuntu@<EC2_PUBLIC_IP>:~/
# OR: git clone https://github.com/your-org/safar-ai.git

# 5. Setup (same as local steps 2–6)
cd safar_ai
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python generate_dataset.py

# 6. Run on all interfaces so it is reachable from the internet
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# Access at: http://<EC2_PUBLIC_IP>:8501
```

To keep it running after you disconnect:

```bash
sudo apt install screen -y
screen -S safar-ai
streamlit run app.py --server.address 0.0.0.0
# Press Ctrl+A then D to detach
# Reconnect: screen -r safar-ai
```

#### Option 2 — Google Cloud Run

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and push Docker image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/safar-ai

# Deploy
gcloud run deploy safar-ai \
  --image gcr.io/YOUR_PROJECT_ID/safar-ai \
  --platform managed \
  --region asia-south1 \
  --port 8501 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated
```

#### Option 3 — Streamlit Community Cloud (free, easiest for demos)

```
1. Push your code to a public GitHub repository
2. Go to https://share.streamlit.io
3. Click "New app"
4. Select your repo, branch = main, file = app.py
5. Click "Deploy"
6. Free public URL provided instantly
```

Note: Remove TensorFlow from requirements.txt for this option (1 GB RAM limit).

---

### D. Edge Device Deployment (Raspberry Pi / Jetson Nano)

This deploys Safar AI as an onboard real-time system inside a highway patrol vehicle.

#### Hardware Required

| Item | Specification | Approx. Cost |
|------|--------------|-------------|
| Raspberry Pi 4 | 4 GB RAM model | ₹5,500 |
| Camera module | Pi Camera v3 or USB webcam | ₹1,500 |
| MicroSD card | 64 GB Class 10 | ₹800 |
| 4G LTE modem | Huawei E3372 USB dongle | ₹2,500 |
| GPS module | u-blox Neo-6M | ₹700 |
| Weatherproof case | IP65 ABS enclosure | ₹600 |
| Power supply | 12V to 5V vehicle USB adapter | ₹400 |
| **Total** | | **~₹12,000** |

#### Step 1 — Flash Raspberry Pi OS

Download the Raspberry Pi Imager from `https://www.raspberrypi.com/software/` and flash Raspberry Pi OS Lite (64-bit) to the SD card. Enable SSH and set a password in the imager settings before flashing.

#### Step 2 — Initial Pi setup

```bash
# SSH into Pi
ssh pi@192.168.1.xxx

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3.11 python3.11-pip python3.11-venv \
     libcamera-apps libglib2.0-0 libgl1 git -y
```

#### Step 3 — Install Safar AI (edge-only build)

```bash
# Copy project to Pi from your laptop
scp -r safar_ai/ pi@192.168.1.xxx:~/

# On the Pi
cd ~/safar_ai
python3.11 -m venv venv && source venv/bin/activate

# Edge-only install (no TF, no Streamlit — saves ~600 MB)
pip install numpy pandas scikit-learn opencv-python-headless \
            scipy statsmodels joblib Pillow
```

#### Step 4 — Run edge inference

Create `edge_runner.py` in the project root:

```python
"""
edge_runner.py — Safar AI Edge Runner
Har Safar, Surakshit Safar

Runs continuously on Raspberry Pi inside a patrol vehicle.
"""
import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from modules.ingestion      import RoadImageIngestor
from modules.spectral       import SpectralReflectivityEngine
from modules.edge_deployment import EdgeInferencePipeline

ingestor = RoadImageIngestor()
engine   = SpectralReflectivityEngine()
pipe     = EdgeInferencePipeline("EDGE_NH48_001", "SEG_LIVE")

print("Safar AI Edge Runner — Har Safar, Surakshit Safar\n")

frame = 0
while True:
    # Replace with actual Pi camera capture:
    # from picamera2 import Picamera2
    # cam = Picamera2(); cam.start()
    # img = cam.capture_array()
    img = ingestor.generate_synthetic_road_image(seed=frame)
    preprocessed, features = ingestor.process_image_full_pipeline(img)

    spectral = engine.analyze_segment("aged_asphalt", "clear", 0.3)
    features["spectral_mean"] = spectral["reflectivity_score"]

    result = pipe.process_frame(features, spectral["reflectivity_score"])
    score  = result["score"]
    status = result["status"]

    print(f"Frame {frame:5d} | Score: {score:.4f} | "
          f"{status['label']:8s} | {result['inference_ms']:.2f}ms")

    if result["alert"]:
        print(f"  *** ALERT: {status['action']} ***")
        # Add LED / buzzer / WhatsApp alert here

    frame += 1
    time.sleep(1)
```

```bash
python edge_runner.py

# To run automatically on boot:
crontab -e
# Add this line:
@reboot /home/pi/safar_ai/venv/bin/python /home/pi/safar_ai/edge_runner.py >> /home/pi/safar_ai/edge.log 2>&1
```

---

### E. Production Deployment Checklist

#### Security
- [ ] Add authentication (Streamlit-Authenticator or Nginx HTTP Basic Auth)
- [ ] Serve over HTTPS (Nginx + Let's Encrypt)
- [ ] Restrict dashboard to NHAI VPN / intranet
- [ ] Rotate edge device IDs and any API keys

#### Reliability
- [ ] Use systemd or supervisor to auto-restart Streamlit
- [ ] Daily backups of `data/` and `models/` to cloud storage
- [ ] Test graceful handling of 4G network outages on edge devices
- [ ] Configure log rotation

#### Performance
- [ ] For more than 100 segments: migrate Digital Twin state to SQLite or PostgreSQL
- [ ] For more than 1,000 daily readings: add Redis cache for dashboard queries
- [ ] Pre-compute ARIMA forecasts nightly via cron instead of on-demand

#### Systemd service (Linux servers)

Save as `/etc/systemd/system/safar-ai.service`:

```ini
[Unit]
Description=Safar AI — Har Safar, Surakshit Safar
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/safar_ai
ExecStart=/home/ubuntu/safar_ai/venv/bin/streamlit run app.py \
          --server.port 8501 --server.address 127.0.0.1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable safar-ai
sudo systemctl start safar-ai
sudo systemctl status safar-ai
```

#### Nginx reverse proxy with HTTPS

```nginx
server {
    listen 443 ssl;
    server_name safar-ai.nhai.gov.in;
    ssl_certificate     /etc/letsencrypt/live/safar-ai.nhai.gov.in/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/safar-ai.nhai.gov.in/privkey.pem;
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

---

## 9. Running the Application

### Quick Reference

| Command | What it does |
|---------|-------------|
| `python generate_dataset.py` | Generate data + train model — run once first |
| `python run_demo.py` | Headless system test, prints full report |
| `streamlit run app.py` | Launch full dashboard |
| `streamlit run app.py --server.port 8080` | Use a different port |
| `streamlit run app.py --server.headless true` | Server / CI mode |
| `python edge_runner.py` | Start edge inference loop on Raspberry Pi |

### First-Time Sequence

```
1.  pip install -r requirements.txt      (5–10 minutes)
2.  python generate_dataset.py           (15 seconds)
3.  python run_demo.py                   (30 seconds, optional)
4.  streamlit run app.py                 (instant)
```

### Subsequent Runs

```
streamlit run app.py
```

Data and model are cached from the first run. No regeneration needed.

---

## 10. Dashboard Guide

### Highway Map tab
- Dot size = proportional to reflectivity score
- Dot color = Green (Good) → Amber (Fair) → Orange (Warning) → Red (Critical)
- Right panel = live alert list with recommended actions

### Reflectivity Monitor tab
- Select any segment from the dropdown
- 180-day history with threshold bands
- Forecast overlay shows next 30 days as a dashed line
- Bottom bar chart compares all 20 segments

### Spectral Analysis tab
- Adjust material, weather, age, dirt, and wear with live controls
- 31-band reflectance curve updates instantly
- Score badge and status update in real time
- Lower panel shows spectral fingerprints of all 7 materials

### Degradation Forecast tab
- Choose segment and forecast horizon (14, 30, or 60 days)
- ARIMA forecast with 90% confidence band
- Red vertical line marks predicted day of Critical threshold breach

### Maintenance Planner tab
- Enter annual budget in crore rupees
- Priority queue sorted by urgency: IMMEDIATE → HIGH → MEDIUM → LOW
- Budget utilization KPI updates live

### AI Explainability tab
- Feature importance bar chart for all 15 features
- Model performance metrics (MAE and R²)
- Correlation heatmap between all features

---

## 11. Configuration Reference

Key constants in `utils.py`:

```python
# Alert thresholds
ALERT_CRITICAL = 0.30    # Below this → CRITICAL (immediate action)
ALERT_WARNING  = 0.50    # Below this → WARNING  (14-day action)
ALERT_GOOD     = 0.70    # Above this → GOOD     (no action)

# Simulation size
SEGMENT_COUNT  = 20      # Number of highway segments to simulate
HISTORY_DAYS   = 180     # Days of synthetic history per segment
FORECAST_DAYS  = 60      # Days ahead to forecast

# Spectral configuration
SPECTRAL_BANDS = np.linspace(400, 700, 31)   # 31 bands at 10 nm steps
```

After changing any value in `utils.py`, re-run `python generate_dataset.py` to rebuild the dataset.

---

## 12. API and Data Formats

### Reflectivity Score
A float in `[0.0, 1.0]`:
- `1.0` = perfect retroreflective surface (new road marking)
- `0.0` = no reflectivity (completely degraded)

### Telemetry Packet (~264 bytes)

```json
{
  "device_id": "EDGE_NH48_001",
  "segment_id": "SEG_001",
  "timestamp": "2024-11-15T08:23:41Z",
  "reflectivity_score": 0.4523,
  "spectral_mean": 0.4810,
  "weather_code": 0,
  "status_code": 2,
  "lat": 28.6139,
  "lon": 77.2090,
  "odometer_km": 42.13,
  "battery_pct": 85,
  "inference_ms": 0.92,
  "flags": 1
}
```

Status codes: 0=Good, 1=Fair, 2=Warning, 3=Critical
Weather codes: 0=Clear, 1=Haze, 2=Rain, 3=Heavy Rain, 4=Fog
Flags bitmask: bit 0 = alert triggered, bit 1 = anomaly detected

---

## 13. Troubleshooting

**`ModuleNotFoundError: No module named 'cv2'`**
```bash
pip install opencv-python-headless
```

**`ModuleNotFoundError: No module named 'tensorflow'`**
```bash
pip install tensorflow
# The system works without TF — GBM + ARIMA path is used automatically
```

**Streamlit shows a blank white page**
```bash
streamlit cache clear
streamlit run app.py
```

**`generate_dataset.py` hangs at ARIMA fitting**

ARIMA can be slow on some systems. Wait up to 60 seconds. If it times out, reduce the segment count:
```bash
# In utils.py, change SEGMENT_COUNT = 20 to SEGMENT_COUNT = 10
# Then re-run:
python generate_dataset.py
```

**Port 8501 already in use**
```bash
streamlit run app.py --server.port 8502
# Or kill the existing process:
lsof -ti:8501 | xargs kill -9
```

**Mapbox map appears blank or grey**

This is normal when offline. The Mapbox scatter map requires an internet connection. All other tabs work fully offline.

**Memory error during training**

In `utils.py`, reduce history length:
```python
HISTORY_DAYS  = 90     # reduce from 180
SEGMENT_COUNT = 10     # reduce from 20
```

---

## 14. Scaling to National Deployment

### Phase 1 — Pilot (Year 1)
Two corridors: NH-48 (Delhi–Mumbai) and NH-44 (Srinagar–Kanyakumari)

| Item | Detail |
|------|--------|
| Edge devices | 40 units (20 per 500 km corridor) |
| Sampling | GPS-triggered every 500m |
| Upload | 4G LTE, delta-encoded, ~264 bytes per reading |
| Estimated CAPEX | ~₹18 lakhs |
| Payback period | Less than 1 month vs accident cost savings |

### Phase 2 — Scale (Year 2–3)
50 high-traffic NH corridors, ~25,000 km

Integrations to add: FASTag transaction counts for wear calibration, IMD weather API for monsoon-aware degradation, IHMCL GNSS tracking for precise GPS, NIC MeghRaj cloud hosting.

### Phase 3 — National Rollout (Year 4–5)
Full 1,46,000 km NHAI network

Projected impact:
- 15,000 accidents per year prevented
- ₹800 crore per year saved through proactive maintenance
- 100% network coverage vs current ~5%
- ₹1,254 per km per year total system cost

---

## 15. Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| Language | Python 3.9–3.11 | All modules |
| Image processing | OpenCV 4.8+ | Camera pipeline, CLAHE |
| Numerical | NumPy, SciPy | Spectral math, smoothing |
| Data | Pandas 2.0+ | DataFrames, time series |
| ML regression | Scikit-learn GBM | Reflectivity prediction |
| Deep learning | TensorFlow/Keras | CNN extractor, LSTM |
| Time series | statsmodels ARIMA | Degradation forecasting |
| Explainability | SHAP | Feature importance |
| Dashboard | Streamlit 1.28+ | Interactive web UI |
| Visualization | Plotly 5.17+ | Charts, Mapbox heatmap |
| Edge inference | NumPy-only INT8 | Quantized on-device model |

---

## 16. License

**MIT License** — Free for NHAI, government use, research, and open-source projects.

---

**Safar AI** — Built for NHAI Hackathon 2024

*हर सफर, सुरक्षित सफर*
*Har Safar, Surakshit Safar*
*Every Journey, A Safe Journey*
