"""
modules/spectral.py — Spectral Reflectivity Engine
===================================================
Simulates hyperspectral reflectance data for road surfaces.
- Models 31 spectral bands from 400–700 nm (visible spectrum)
- Generates realistic reflectance curves for different road materials
- Applies environmental noise (weather, dirt, wear, aging)
- Computes a single weighted Reflectivity Score (0–1)
- Provides spectral fingerprinting for material classification

Scientific Basis:
  Road markings (retroreflective paint) have high reflectance across
  400–700 nm. Aged asphalt absorbs most visible light (low R).
  Moisture reduces R significantly; dirt broadens the absorption curve.

Author: Safar AI Team
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Dict, List, Optional, Tuple
import logging

from utils import SPECTRAL_BANDS, WEATHER_FACTORS, SURFACE_BASE_REFLECTIVITY, noise, clamp

logger = logging.getLogger("SafarAI.spectral")


# ─────────────────────────────────────────────
# Spectral Signature Library
# ─────────────────────────────────────────────

def _gaussian(x: np.ndarray, mu: float, sigma: float, amp: float) -> np.ndarray:
    """Gaussian peak for spectral feature modeling."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def get_spectral_signature(material: str, age_factor: float = 0.0) -> np.ndarray:
    """
    Return a base spectral reflectance curve for a given road material.

    Args:
        material: One of the keys in SURFACE_BASE_REFLECTIVITY.
        age_factor: 0 (new) → 1 (completely degraded). Scales reflectance down.

    Returns:
        ndarray of shape (31,) — reflectance per band in [0, 1].
    """
    bands = SPECTRAL_BANDS
    n     = len(bands)
    base  = SURFACE_BASE_REFLECTIVITY.get(material, 0.65)

    if "road_marking" in material:
        # Retroreflective paint: broad high reflectance, small dip near 580 nm
        curve = np.ones(n) * base
        curve -= _gaussian(bands, 580, 20, 0.12)   # slight yellow dip
        curve += _gaussian(bands, 450, 25, 0.05)   # blue enhancement

    elif "concrete" in material:
        # Concrete: flat mid-grey, slight rise toward red end
        curve = np.ones(n) * base
        curve += np.linspace(-0.02, 0.06, n)       # gradual red slope

    elif "new_asphalt" in material:
        # Fresh asphalt: dark but with retroreflective aggregates
        curve = np.ones(n) * base
        curve -= _gaussian(bands, 450, 40, 0.08)   # blue absorption
        curve += _gaussian(bands, 650, 30, 0.06)   # slight red peak (aggregates)

    elif "aged_asphalt" in material:
        curve = np.ones(n) * base
        curve -= np.linspace(0.0, 0.10, n)         # overall decay toward red
        curve += noise(n, 0.015)

    elif "worn_asphalt" in material:
        curve = np.ones(n) * base
        curve -= _gaussian(bands, 500, 60, 0.15)   # broad green-yellow absorption

    elif "cracked_asphalt" in material:
        curve = np.ones(n) * base
        # Deep cracks create shadow regions → strong scattering loss
        curve *= 0.85
        curve += noise(n, 0.03)

    else:
        curve = np.ones(n) * base

    # Apply aging degradation
    degradation = 1.0 - age_factor * 0.6   # max 60% reduction
    curve = curve * degradation

    # Smooth the curve (real spectrometers always produce smooth outputs)
    curve = savgol_filter(np.clip(curve, 0, 1), window_length=7, polyorder=2)
    return np.clip(curve, 0.0, 1.0)


# ─────────────────────────────────────────────
# Noise & Environmental Effects
# ─────────────────────────────────────────────

def apply_environmental_noise(
    curve: np.ndarray,
    weather: str = "clear",
    dirt_level: float = 0.0,
    wear_level: float = 0.0,
) -> np.ndarray:
    """
    Apply real-world noise and environmental effects to a spectral curve.

    Args:
        curve: Base spectral reflectance array (31,).
        weather: Weather condition string.
        dirt_level: 0–1, how dirty the surface is.
        wear_level: 0–1, physical wear/abrasion level.

    Returns:
        Modified reflectance curve (31,).
    """
    curve = curve.copy()
    bands = SPECTRAL_BANDS

    # Weather effect: moisture reduces reflectance (especially in blue-green)
    weather_mult = WEATHER_FACTORS.get(weather, 1.0)
    if weather in ("rain", "heavy_rain"):
        # Water film: suppresses R in visible, adds glare specular component
        moisture_suppression = _gaussian(bands, 520, 80, 0.20 * (weather_mult - 1.0))
        curve -= moisture_suppression
        # Add specular glare at angles (simplified as uniform lift in red)
        curve += _gaussian(bands, 650, 30, 0.05)

    elif weather == "fog":
        # Scattering: broadband reduction + diffuse boost
        curve *= 0.80
        curve += noise(len(bands), 0.02)

    elif weather == "haze":
        curve *= 0.88
        curve += _gaussian(bands, 440, 30, 0.03)   # bluish haze lift

    # Dirt effect: broadband absorption (dirt/dust is spectrally flat)
    if dirt_level > 0:
        dirt_suppression = dirt_level * 0.35 * np.ones(len(bands))
        dirt_suppression += _gaussian(bands, 580, 50, dirt_level * 0.10)  # ochre dirt peak
        curve -= dirt_suppression

    # Wear effect: loss of retroreflective microspheres
    if wear_level > 0:
        # Microspheres are most effective in 450–550 nm range
        wear_loss = _gaussian(bands, 500, 60, wear_level * 0.30)
        curve -= wear_loss

    # Instrument noise (simulating spectrometer noise floor)
    curve += noise(len(bands), 0.008)

    return np.clip(curve, 0.0, 1.0)


# ─────────────────────────────────────────────
# Reflectivity Score Computation
# ─────────────────────────────────────────────

# Perceptual luminosity weights (how human drivers perceive brightness)
# Higher sensitivity in green (550 nm), lower at extremes
_PERCEPTUAL_WEIGHTS = np.array([
    0.03, 0.04, 0.05, 0.06, 0.08,   # 400–440 nm (violet-blue)
    0.10, 0.12, 0.14, 0.16, 0.14,   # 450–490 nm (blue-cyan)
    0.12, 0.14, 0.18, 0.22, 0.26,   # 500–540 nm (cyan-green)
    0.30, 0.35, 0.38, 0.35, 0.30,   # 550–590 nm (green-yellow) ← peak
    0.25, 0.20, 0.16, 0.12, 0.10,   # 600–640 nm (orange-red)
    0.08, 0.06, 0.05, 0.04, 0.03,   # 650–690 nm (deep red)
    0.03,                             # 700 nm
])
_PERCEPTUAL_WEIGHTS = _PERCEPTUAL_WEIGHTS / _PERCEPTUAL_WEIGHTS.sum()  # normalize


def compute_reflectivity_score(
    curve: np.ndarray,
    visual_features: Optional[Dict] = None
) -> float:
    """
    Compute a single Reflectivity Score [0–1] from a spectral curve.
    Fuses spectral data with optional visual features for higher accuracy.

    Args:
        curve: Spectral reflectance array (31,).
        visual_features: Dict from ingestion.extract_visual_features() [optional].

    Returns:
        Reflectivity score in [0, 1].
    """
    # Weighted spectral integral (luminosity-weighted)
    spectral_score = float(np.dot(curve, _PERCEPTUAL_WEIGHTS))

    if visual_features:
        # Fuse with visual features (brightness, bright_ratio)
        vis_score = (
            0.40 * visual_features.get("mean_brightness", spectral_score) +
            0.30 * visual_features.get("bright_ratio",    0.0) +
            0.20 * visual_features.get("michelson_contrast", 0.5) +
            0.10 * min(visual_features.get("laplacian_var", 0.5), 1.0)
        )
        # Fuse spectral (70%) + visual (30%)
        final_score = 0.70 * spectral_score + 0.30 * vis_score
    else:
        final_score = spectral_score

    return clamp(final_score)


# ─────────────────────────────────────────────
# Spectral Reflectivity Engine (Main Class)
# ─────────────────────────────────────────────

class SpectralReflectivityEngine:
    """
    Main engine for hyperspectral road surface analysis.
    Generates spectral data, applies noise, and outputs reflectivity scores.

    Usage:
        engine = SpectralReflectivityEngine()
        result = engine.analyze_segment(material='new_asphalt', weather='rain', age=0.3)
        print(result['reflectivity_score'])
    """

    def __init__(self):
        self.bands = SPECTRAL_BANDS
        logger.info("SpectralReflectivityEngine initialized | bands=%d", len(self.bands))

    def analyze_segment(
        self,
        material: str = "new_asphalt",
        weather: str = "clear",
        age_factor: float = 0.0,
        dirt_level: float = 0.0,
        wear_level: float = 0.0,
        visual_features: Optional[Dict] = None,
    ) -> Dict:
        """
        Full spectral analysis of one highway segment.

        Args:
            material: Road surface material.
            weather: Weather condition.
            age_factor: 0–1 aging level.
            dirt_level: 0–1 dirt contamination.
            wear_level: 0–1 physical wear.
            visual_features: Optional dict from ingestion module.

        Returns:
            Dict with keys: 'bands', 'raw_curve', 'noisy_curve',
                            'reflectivity_score', 'material', 'condition_params'.
        """
        raw_curve   = get_spectral_signature(material, age_factor)
        noisy_curve = apply_environmental_noise(raw_curve, weather, dirt_level, wear_level)
        score       = compute_reflectivity_score(noisy_curve, visual_features)

        return {
            "bands":              self.bands.tolist(),
            "raw_curve":          raw_curve.tolist(),
            "noisy_curve":        noisy_curve.tolist(),
            "reflectivity_score": score,
            "material":           material,
            "condition_params": {
                "weather":     weather,
                "age_factor":  age_factor,
                "dirt_level":  dirt_level,
                "wear_level":  wear_level,
            },
        }

    def batch_analyze(
        self,
        segments: List[Dict]
    ) -> pd.DataFrame:
        """
        Analyze multiple segments at once.

        Args:
            segments: List of dicts, each with keys matching analyze_segment args.

        Returns:
            DataFrame with one row per segment.
        """
        records = []
        for i, seg in enumerate(segments):
            result = self.analyze_segment(**{
                k: seg[k] for k in
                ("material","weather","age_factor","dirt_level","wear_level")
                if k in seg
            })
            records.append({
                "segment_idx":         i,
                "material":            result["material"],
                "weather":             result["condition_params"]["weather"],
                "age_factor":          result["condition_params"]["age_factor"],
                "reflectivity_score":  result["reflectivity_score"],
                "spectral_mean":       float(np.mean(result["noisy_curve"])),
                "spectral_std":        float(np.std(result["noisy_curve"])),
            })
        return pd.DataFrame(records)

    def generate_synthetic_dataset(
        self,
        n_segments: int = 20,
        n_timesteps: int = 180,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate a synthetic time-series spectral dataset for n_segments
        over n_timesteps days, simulating progressive degradation.

        Args:
            n_segments: Number of highway segments.
            n_timesteps: Number of days to simulate.
            seed: Random seed.

        Returns:
            DataFrame with columns: segment_id, timestep, date, material,
                                    weather, age_factor, reflectivity_score,
                                    spectral_mean, dirt_level, wear_level.
        """
        np.random.seed(seed)
        materials  = list(SURFACE_BASE_REFLECTIVITY.keys())
        weathers   = list(WEATHER_FACTORS.keys())
        records    = []

        # Assign each segment a random starting material and degradation rate
        seg_materials     = np.random.choice(
            ["new_asphalt","aged_asphalt","road_marking_new","concrete"], n_segments
        )
        seg_degrade_rates = np.random.uniform(0.001, 0.006, n_segments)  # per day
        seg_dirt_rates    = np.random.uniform(0.0005, 0.003, n_segments)
        seg_wear_rates    = np.random.uniform(0.0005, 0.004, n_segments)

        import pandas as _pd
        from utils import timestamp_range
        dates = list(timestamp_range(n_timesteps))

        weather_sequence = np.random.choice(weathers, n_timesteps,
                                            p=[0.55, 0.15, 0.15, 0.08, 0.07])

        for s in range(n_segments):
            seg_id   = f"SEG_{s+1:03d}"
            material = seg_materials[s]

            for t in range(n_timesteps):
                age    = clamp(t * seg_degrade_rates[s])
                dirt   = clamp(t * seg_dirt_rates[s])
                wear   = clamp(t * seg_wear_rates[s])
                # Seasonal effect (India: monsoon in June-Sept)
                month  = dates[t].month
                weather = weather_sequence[t]
                if month in (6, 7, 8, 9):  # monsoon months
                    weather = np.random.choice(["rain","heavy_rain","haze"],
                                               p=[0.4, 0.3, 0.3])

                result = self.analyze_segment(material, weather, age, dirt, wear)
                records.append({
                    "segment_id":         seg_id,
                    "timestep":           t,
                    "date":               dates[t],
                    "material":           material,
                    "weather":            weather,
                    "age_factor":         round(age, 4),
                    "dirt_level":         round(dirt, 4),
                    "wear_level":         round(wear, 4),
                    "reflectivity_score": round(result["reflectivity_score"], 4),
                    "spectral_mean":      round(result.get("spectral_mean",
                                                float(np.mean(result["noisy_curve"]))), 4),
                })

        df = _pd.DataFrame(records)
        logger.info("Synthetic dataset generated: %d rows", len(df))
        return df
