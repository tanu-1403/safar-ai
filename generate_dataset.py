"""
generate_dataset.py — Standalone Dataset & Model Generator
============================================================
Run this FIRST before launching the dashboard.
Generates synthetic data, trains the model, saves all artifacts.

Usage:
    cd safar_ai
    python generate_dataset.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.spectral    import SpectralReflectivityEngine
from modules.model       import ReflectivityPredictor, build_training_dataset
from modules.digital_twin import build_digital_twin_from_dataset
from modules.prediction  import PredictionPipeline
from utils import ensure_dir

print("\n" + "═"*60)
print("  Safar AI — Dataset & Model Generator")
print("═"*60 + "\n")

# ── Config ────────────────────────────────────────────────────────────────────
N_SEGMENTS  = 20
N_DAYS      = 180
SEED        = 42

ensure_dir("data")
ensure_dir("models")

# ── Step 1: Generate Synthetic Spectral Dataset ────────────────────────────
print("📡 [1/5] Generating synthetic hyperspectral dataset...")
engine = SpectralReflectivityEngine()
df     = engine.generate_synthetic_dataset(N_SEGMENTS, N_DAYS, SEED)
df.to_csv("data/synthetic_spectral_data.csv", index=False)
print(f"   ✅ Dataset: {len(df):,} rows × {len(df.columns)} columns → data/synthetic_spectral_data.csv\n")

# ── Step 2: Build Training Features ──────────────────────────────────────────
print("🔧 [2/5] Engineering training features (visual + spectral + environmental)...")
train_df = build_training_dataset(df)
train_df.to_csv("data/training_features.csv", index=False)
print(f"   ✅ Training dataset: {len(train_df):,} rows × {len(train_df.columns)} columns\n")

# ── Step 3: Train AI Model ────────────────────────────────────────────────────
print("🤖 [3/5] Training Gradient Boosting Reflectivity Predictor...")
predictor = ReflectivityPredictor()
metrics   = predictor.train(train_df)
predictor.save()
print(f"   ✅ Training complete:")
print(f"      Train MAE: {metrics['train_mae']} | Train R²: {metrics['train_r2']}")
print(f"      Test  MAE: {metrics['test_mae']}  | Test  R²: {metrics['test_r2']}\n")

# ── Step 4: Build Digital Twin ────────────────────────────────────────────────
print("🏗️  [4/5] Constructing Digital Twin registry...")
registry = build_digital_twin_from_dataset(df, n_segments=N_SEGMENTS)
registry.save()
n_segs   = len(registry.all_segments())
n_alerts = len(registry.get_alerts())
print(f"   ✅ Digital Twin: {n_segs} segments created")
print(f"   ⚠️  Alerts requiring action: {n_alerts}\n")

# ── Step 5: Fit Prediction Pipeline ──────────────────────────────────────────
print("📈 [5/5] Fitting ARIMA forecasters for all segments...")
hist_df  = registry.history_dataframe()
pipeline = PredictionPipeline(use_lstm=False)
pipeline.fit_all(hist_df)
forecasts = pipeline.forecast_all(hist_df, steps=60)

# Save summary report
summary_df = registry.summary_dataframe()
summary_df.to_csv("data/segment_summary.csv", index=False)

alerts_df = registry.get_alerts()
alerts_df.to_csv("data/active_alerts.csv", index=False)

recs = pipeline.get_recommendations(
    summary_df,
    {row["segment_id"]: max(1, row["end_km"] - row["start_km"]) for _, row in summary_df.iterrows()}
)
recs.to_csv("data/maintenance_recommendations.csv", index=False)

print(f"   ✅ Forecasts computed for {len(forecasts)} segments\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print("═"*60)
print("  GENERATION COMPLETE — Summary")
print("═"*60)
print(f"  📁 data/synthetic_spectral_data.csv      {len(df):>8,} rows")
print(f"  📁 data/training_features.csv            {len(train_df):>8,} rows")
print(f"  📁 data/segment_summary.csv              {len(summary_df):>8,} rows")
print(f"  📁 data/active_alerts.csv                {len(alerts_df):>8,} rows")
print(f"  📁 data/maintenance_recommendations.csv  {len(recs):>8,} rows")
print(f"  🤖 models/reflectivity_predictor.joblib")
print(f"  🏗️  data/digital_twin_state.json")
print()

# Segment status breakdown
status_counts = summary_df["status_label"].value_counts()
print("  Segment Status Distribution:")
for status, count in status_counts.items():
    icons = {"GOOD":"✅","FAIR":"⚠️","WARNING":"🔶","CRITICAL":"🚨"}
    bar = "█" * count
    print(f"    {icons.get(status,'❓')} {status:<10} {bar} ({count})")

print()
print("  Next step: streamlit run app.py")
print("═"*60 + "\n")
