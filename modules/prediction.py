"""
modules/prediction.py — Prediction & Degradation Engine
=========================================================
Forecasts future reflectivity degradation for each segment.

Two models are implemented:
  1. ARIMA — fast, interpretable, great for smooth time series
  2. LSTM  — captures seasonal/non-linear patterns (requires TF)

Also provides:
  - Maintenance scheduling recommender
  - Priority ranking (which segments need attention first)
  - Cost-benefit analysis for proactive vs reactive maintenance

Author: Safar AI Team
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, List, Optional, Tuple

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# LSTM (TF optional)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from utils import (
    ALERT_CRITICAL, ALERT_WARNING, ALERT_GOOD,
    FORECAST_DAYS, reflectivity_to_status, clamp
)

logger = logging.getLogger("SafarAI.prediction")
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────
# ARIMA Forecaster
# ─────────────────────────────────────────────

class ARIMAForecaster:
    """
    ARIMA-based reflectivity time-series forecaster.

    Automatically selects differencing order (d) using ADF test.
    Uses order (2, d, 2) as a robust default for degradation curves.
    """

    def __init__(self, order: Tuple = (2, 1, 2)):
        self.order   = order
        self.models  = {}   # segment_id → fitted model
        self.history = {}   # segment_id → training series

    def _check_stationarity(self, series: np.ndarray) -> int:
        """Return differencing order needed for stationarity."""
        try:
            adf_result = adfuller(series, autolag="AIC")
            p_value    = adf_result[1]
            return 0 if p_value < 0.05 else 1
        except Exception:
            return 1

    def fit(self, segment_id: str, series: np.ndarray):
        """
        Fit ARIMA model to a reflectivity time series.

        Args:
            segment_id: Identifier for this segment's model.
            series: 1D array of reflectivity scores (chronological).
        """
        if len(series) < 10:
            logger.warning("Series too short for ARIMA fit: %s (%d pts)", segment_id, len(series))
            self.models[segment_id]  = None
            self.history[segment_id] = series
            return

        d = self._check_stationarity(series)
        order = (self.order[0], d, self.order[2])

        try:
            fitted = ARIMA(series, order=order).fit()
            self.models[segment_id]  = fitted
            self.history[segment_id] = series
        except Exception as e:
            logger.warning("ARIMA fit failed for %s: %s", segment_id, e)
            self.models[segment_id]  = None
            self.history[segment_id] = series

    def forecast(self, segment_id: str, steps: int = FORECAST_DAYS) -> np.ndarray:
        """
        Forecast future reflectivity scores.

        Args:
            segment_id: Segment to forecast.
            steps: Number of days ahead.

        Returns:
            1D array of length `steps`, values in [0, 1].
        """
        model = self.models.get(segment_id)
        if model is None:
            # Fallback: linear extrapolation from last known values
            hist = self.history.get(segment_id, np.array([0.5]))
            if len(hist) >= 5:
                slope = np.polyfit(range(len(hist[-20:])), hist[-20:], 1)[0]
                last  = hist[-1]
                fc    = [clamp(last + slope * i) for i in range(1, steps + 1)]
            else:
                fc = [clamp(hist[-1] - 0.002 * i) for i in range(1, steps + 1)]
            return np.array(fc)

        try:
            fc = model.forecast(steps=steps)
            return np.clip(fc, 0.0, 1.0)
        except Exception as e:
            logger.warning("ARIMA forecast failed for %s: %s", segment_id, e)
            hist  = self.history.get(segment_id, np.array([0.5]))
            slope = np.polyfit(range(len(hist[-20:])), hist[-20:], 1)[0] if len(hist) >= 5 else -0.002
            return np.array([clamp(hist[-1] + slope * i) for i in range(1, steps + 1)])

    def forecast_with_confidence(
        self,
        segment_id: str,
        steps: int = FORECAST_DAYS,
        alpha: float = 0.10
    ) -> Dict:
        """
        Forecast with confidence intervals.

        Returns:
            Dict with 'mean', 'lower', 'upper' arrays.
        """
        model = self.models.get(segment_id)
        mean  = self.forecast(segment_id, steps)

        if model is None:
            # Synthetic CI from linear extrapolation
            uncertainty = np.linspace(0.02, 0.08, steps)
            return {
                "mean":  mean,
                "lower": np.clip(mean - uncertainty, 0, 1),
                "upper": np.clip(mean + uncertainty, 0, 1),
            }

        try:
            fc_obj = model.get_forecast(steps=steps)
            ci     = fc_obj.conf_int(alpha=alpha)
            return {
                "mean":  np.clip(fc_obj.predicted_mean, 0, 1),
                "lower": np.clip(ci.iloc[:, 0].values, 0, 1),
                "upper": np.clip(ci.iloc[:, 1].values, 0, 1),
            }
        except Exception:
            uncertainty = np.linspace(0.02, 0.08, steps)
            return {
                "mean":  mean,
                "lower": np.clip(mean - uncertainty, 0, 1),
                "upper": np.clip(mean + uncertainty, 0, 1),
            }


# ─────────────────────────────────────────────
# LSTM Forecaster (Deep Learning)
# ─────────────────────────────────────────────

def build_lstm_model(lookback: int = 30, n_features: int = 1) -> Optional[object]:
    """
    Build a lightweight LSTM model for reflectivity sequence forecasting.

    Architecture:
      LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(16) → Dense(1)

    Args:
        lookback: Input sequence length (days).
        n_features: Number of input features per timestep.

    Returns:
        Compiled Keras Model or None.
    """
    if not TF_AVAILABLE:
        return None

    inp = Input(shape=(lookback, n_features), name="sequence_input")
    x   = layers.LSTM(64, return_sequences=True, name="lstm_1")(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.LSTM(32, return_sequences=False, name="lstm_2")(x)
    x   = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1,  activation="sigmoid", name="reflectivity_output")(x)

    model = Model(inputs=inp, outputs=out, name="LSTMReflectivityForecaster")
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
    return model


class LSTMForecaster:
    """
    LSTM-based multi-step forecaster.
    Trains one shared model across all segments (transfer learning style).
    """

    LOOKBACK = 30   # days of history used as input

    def __init__(self):
        self.model   = build_lstm_model(self.LOOKBACK)
        self.trained = False

    def _prepare_sequences(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a time series to supervised (X, y) sequences."""
        X, y = [], []
        for i in range(len(series) - self.LOOKBACK):
            X.append(series[i:i + self.LOOKBACK])
            y.append(series[i + self.LOOKBACK])
        return np.array(X)[..., np.newaxis], np.array(y)

    def fit(self, all_series: List[np.ndarray], epochs: int = 30, verbose: int = 0):
        """
        Train LSTM on multiple segment series (pooled training).

        Args:
            all_series: List of 1D numpy arrays (one per segment).
            epochs: Training epochs.
        """
        if self.model is None:
            logger.warning("TF unavailable — LSTM training skipped.")
            return

        X_all, y_all = [], []
        for series in all_series:
            if len(series) > self.LOOKBACK + 5:
                X, y = self._prepare_sequences(series)
                X_all.append(X)
                y_all.append(y)

        if not X_all:
            return

        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)

        logger.info("LSTM training | samples=%d, epochs=%d", len(X_all), epochs)
        self.model.fit(
            X_all, y_all,
            epochs=epochs,
            batch_size=64,
            validation_split=0.1,
            verbose=verbose,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        self.trained = True
        logger.info("LSTM training complete.")

    def forecast(self, series: np.ndarray, steps: int = FORECAST_DAYS) -> np.ndarray:
        """
        Multi-step forecast by iterative prediction (auto-regressive).

        Args:
            series: Historical reflectivity array (must be ≥ LOOKBACK length).
            steps: Number of days to forecast.

        Returns:
            1D numpy array of length `steps`.
        """
        if self.model is None or not self.trained:
            # Fallback to simple linear trend
            if len(series) >= 10:
                slope = np.polyfit(range(len(series[-20:])), series[-20:], 1)[0]
                return np.array([clamp(series[-1] + slope * i) for i in range(1, steps + 1)])
            return np.full(steps, series[-1] if len(series) else 0.5)

        window = series[-self.LOOKBACK:].copy()
        preds  = []

        for _ in range(steps):
            x    = window[np.newaxis, :, np.newaxis]  # (1, lookback, 1)
            pred = float(self.model.predict(x, verbose=0)[0, 0])
            preds.append(clamp(pred))
            window = np.roll(window, -1)
            window[-1] = pred

        return np.array(preds)


# ─────────────────────────────────────────────
# Maintenance Scheduler
# ─────────────────────────────────────────────

class MaintenanceScheduler:
    """
    Recommends maintenance actions and schedules based on
    current reflectivity, forecast trajectories, and cost models.
    """

    # Cost models (INR per km per event)
    COST_MODELS = {
        "repainting":    {"cost_per_km": 25000,  "reflectivity_boost": 0.45, "duration_days": 3},
        "resurfacing":   {"cost_per_km": 350000, "reflectivity_boost": 0.70, "duration_days": 14},
        "cleaning":      {"cost_per_km": 3500,   "reflectivity_boost": 0.10, "duration_days": 1},
        "inspection":    {"cost_per_km": 1500,   "reflectivity_boost": 0.00, "duration_days": 1},
        "microsurfacing":{"cost_per_km": 80000,  "reflectivity_boost": 0.35, "duration_days": 5},
    }

    # Days until estimated threshold breach → urgency bucket
    URGENCY_MAP = {
        (0,  7):  "IMMEDIATE",
        (7,  30): "HIGH",
        (30, 60): "MEDIUM",
        (60, 999):"LOW",
    }

    def recommend(
        self,
        segment_summary: pd.DataFrame,
        forecast_dict: Dict[str, np.ndarray],
        segment_lengths_km: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Generate a prioritized maintenance recommendation report.

        Args:
            segment_summary: Summary DataFrame from DigitalTwinRegistry.
            forecast_dict: {segment_id → forecast array}.
            segment_lengths_km: {segment_id → length in km}.

        Returns:
            DataFrame with maintenance recommendations.
        """
        rows = []
        for _, row in segment_summary.iterrows():
            sid   = row["segment_id"]
            score = row.get("reflectivity_score", 0.5)
            if score is None:
                continue

            fc    = forecast_dict.get(sid, np.array([score]))
            # Days until CRITICAL threshold
            days_to_critical = next(
                (i + 1 for i, v in enumerate(fc) if v < ALERT_CRITICAL),
                len(fc) + 1
            )
            # Days until WARNING threshold
            days_to_warning  = next(
                (i + 1 for i, v in enumerate(fc) if v < ALERT_WARNING),
                len(fc) + 1
            ) if score >= ALERT_WARNING else 0

            # Determine urgency
            urgency = "LOW"
            for (lo, hi), label in self.URGENCY_MAP.items():
                if lo <= days_to_critical < hi:
                    urgency = label
                    break

            # Recommend action
            if score < ALERT_CRITICAL:
                action = "resurfacing"
            elif score < ALERT_WARNING:
                action = "repainting"
            elif score < ALERT_GOOD:
                action = "microsurfacing"
            else:
                action = "inspection"

            km      = segment_lengths_km.get(sid, 10.0)
            cost    = self.COST_MODELS[action]["cost_per_km"] * km
            boost   = self.COST_MODELS[action]["reflectivity_boost"]
            post_r  = clamp(score + boost)

            rows.append({
                "segment_id":         sid,
                "current_score":      round(score, 3),
                "recommended_action": action,
                "urgency":            urgency,
                "days_to_critical":   days_to_critical,
                "days_to_warning":    days_to_warning,
                "post_maintenance_r": round(post_r, 3),
                "estimated_cost_inr": round(cost, 0),
                "segment_km":         round(km, 1),
            })

        df = pd.DataFrame(rows)
        if len(df):
            urgency_order = {"IMMEDIATE": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            df["_sort"] = df["urgency"].map(urgency_order)
            df = df.sort_values(["_sort", "current_score"]).drop(columns=["_sort"])
        return df

    def budget_analysis(
        self,
        recommendations: pd.DataFrame,
        annual_budget_inr: float = 50_000_000
    ) -> Dict:
        """
        Compute budget allocation and ROI metrics.

        Returns:
            Dict with total cost, within-budget segments, prioritization summary.
        """
        if len(recommendations) == 0:
            return {}

        total_cost = recommendations["estimated_cost_inr"].sum()
        immediate  = recommendations[recommendations["urgency"] == "IMMEDIATE"]["estimated_cost_inr"].sum()
        high       = recommendations[recommendations["urgency"] == "HIGH"]["estimated_cost_inr"].sum()

        return {
            "total_recommended_cost":     round(total_cost, 0),
            "immediate_spend_required":   round(immediate, 0),
            "high_priority_spend":        round(high, 0),
            "annual_budget":              annual_budget_inr,
            "budget_utilization_pct":     round(min(total_cost / annual_budget_inr * 100, 100), 1),
            "segments_in_immediate":      int((recommendations["urgency"] == "IMMEDIATE").sum()),
            "segments_in_high":           int((recommendations["urgency"] == "HIGH").sum()),
            "avg_post_maintenance_score": round(recommendations["post_maintenance_r"].mean(), 3),
        }


# ─────────────────────────────────────────────
# Prediction Pipeline (Convenience Wrapper)
# ─────────────────────────────────────────────

class PredictionPipeline:
    """
    High-level prediction pipeline:
      1. Fits ARIMA to each segment's history
      2. Optionally trains LSTM (if TF available)
      3. Generates forecasts + CI for all segments
      4. Runs maintenance scheduler
    """

    def __init__(self, use_lstm: bool = False):
        self.arima     = ARIMAForecaster()
        self.lstm      = LSTMForecaster() if (use_lstm and TF_AVAILABLE) else None
        self.scheduler = MaintenanceScheduler()
        self.forecasts: Dict[str, Dict] = {}

    def fit_all(self, history_df: pd.DataFrame):
        """
        Fit models to all segments in the history DataFrame.

        Args:
            history_df: From DigitalTwinRegistry.history_dataframe()
        """
        seg_ids = history_df["segment_id"].unique()
        all_series = []

        for sid in seg_ids:
            seg_data = history_df[history_df["segment_id"] == sid].sort_values("timestamp")
            series   = seg_data["reflectivity_score"].values
            self.arima.fit(sid, series)
            all_series.append(series)

        if self.lstm:
            self.lstm.fit(all_series, epochs=20)

        logger.info("PredictionPipeline fitted | %d segments", len(seg_ids))

    def forecast_all(
        self,
        history_df: pd.DataFrame,
        steps: int = FORECAST_DAYS
    ) -> Dict[str, Dict]:
        """
        Generate forecasts for all segments.

        Returns:
            Dict: {segment_id → {'mean': array, 'lower': array, 'upper': array}}
        """
        seg_ids = history_df["segment_id"].unique()

        for sid in seg_ids:
            arima_fc = self.arima.forecast_with_confidence(sid, steps)

            if self.lstm and self.lstm.trained:
                seg_data = history_df[history_df["segment_id"] == sid].sort_values("timestamp")
                series   = seg_data["reflectivity_score"].values
                lstm_fc  = self.lstm.forecast(series, steps)
                # Ensemble: 60% ARIMA + 40% LSTM
                mean = 0.6 * arima_fc["mean"] + 0.4 * lstm_fc
                span = arima_fc["upper"] - arima_fc["lower"]
                self.forecasts[sid] = {
                    "mean":  np.clip(mean, 0, 1),
                    "lower": np.clip(mean - span/2, 0, 1),
                    "upper": np.clip(mean + span/2, 0, 1),
                }
            else:
                self.forecasts[sid] = arima_fc

        return self.forecasts

    def get_recommendations(
        self,
        summary_df: pd.DataFrame,
        segment_lengths: Dict[str, float]
    ) -> pd.DataFrame:
        """Generate maintenance recommendations using current forecasts."""
        return self.scheduler.recommend(
            summary_df,
            {sid: fc["mean"] for sid, fc in self.forecasts.items()},
            segment_lengths,
        )
