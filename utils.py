"""
utils.py — Shared utility functions for Safar AI System.
Contains logging, config constants, color maps, alert thresholds,
and helper routines used across all modules.
"""

import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime

# ─────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SafarAI")


# ─────────────────────────────────────────────
# System-wide Constants
# ─────────────────────────────────────────────
SPECTRAL_BANDS = np.linspace(400, 700, 31)   # 31 bands from 400–700 nm (10 nm steps)
SEGMENT_COUNT  = 20                            # Default highway segments in simulation
HISTORY_DAYS   = 180                           # Days of historical data to simulate
FORECAST_DAYS  = 60                            # Days ahead to predict
ALERT_CRITICAL = 0.30                          # Reflectivity below this → CRITICAL alert
ALERT_WARNING  = 0.50                          # Reflectivity below this → WARNING alert
ALERT_GOOD     = 0.70                          # Reflectivity above this → GOOD (no action)

# NH corridor metadata (simulated, representative)
NH_CORRIDORS = {
    "NH-48 (Delhi–Mumbai)":     {"start_km": 0,   "end_km": 1400, "state": "Multi-State"},
    "NH-44 (Srinagar–Kanniya)": {"start_km": 0,   "end_km": 3800, "state": "Multi-State"},
    "NH-19 (Delhi–Kolkata)":    {"start_km": 0,   "end_km": 1453, "state": "Multi-State"},
    "NH-8 (Delhi–Jaipur)":      {"start_km": 0,   "end_km": 290,  "state": "Rajasthan"},
    "NH-4 (Mumbai–Chennai)":    {"start_km": 0,   "end_km": 1235, "state": "Multi-State"},
}

# Weather condition → degradation multiplier
WEATHER_FACTORS = {
    "clear":    1.00,
    "haze":     1.10,
    "rain":     1.25,
    "heavy_rain": 1.45,
    "fog":      1.15,
}

# Road surface type → base reflectivity
SURFACE_BASE_REFLECTIVITY = {
    "new_asphalt":       0.88,
    "aged_asphalt":      0.65,
    "worn_asphalt":      0.42,
    "cracked_asphalt":   0.30,
    "road_marking_new":  0.95,
    "road_marking_faded":0.40,
    "concrete":          0.75,
}


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def reflectivity_to_status(score: float) -> dict:
    """
    Convert a numeric reflectivity score to a status label and color.

    Args:
        score: Reflectivity value in [0, 1].

    Returns:
        dict with keys 'label', 'color', 'emoji', 'action'.
    """
    if score >= ALERT_GOOD:
        return {"label": "GOOD",     "color": "#22c55e", "emoji": "✅", "action": "No action required"}
    elif score >= ALERT_WARNING:
        return {"label": "FAIR",     "color": "#f59e0b", "emoji": "⚠️", "action": "Schedule inspection within 30 days"}
    elif score >= ALERT_CRITICAL:
        return {"label": "WARNING",  "color": "#f97316", "emoji": "🔶", "action": "Prioritize maintenance within 14 days"}
    else:
        return {"label": "CRITICAL", "color": "#ef4444", "emoji": "🚨", "action": "IMMEDIATE maintenance required"}


def generate_segment_ids(n: int = SEGMENT_COUNT, corridor: str = "NH-48") -> list:
    """Generate segment ID strings like 'NH-48_SEG_001'."""
    return [f"{corridor}_SEG_{i+1:03d}" for i in range(n)]


def timestamp_range(days: int = HISTORY_DAYS, freq: str = "D") -> pd.DatetimeIndex:
    """Return a pandas DatetimeIndex ending today, spanning `days` days."""
    end   = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=days - 1)
    return pd.date_range(start=start, end=end, freq=freq)


def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, val))


def smooth(series: np.ndarray, window: int = 7) -> np.ndarray:
    """Apply a simple moving-average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(series, kernel, mode="same")


def noise(shape, scale: float = 0.02) -> np.ndarray:
    """Return Gaussian noise array of given shape."""
    return np.random.normal(0, scale, shape)


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, return path."""
    os.makedirs(path, exist_ok=True)
    return path
