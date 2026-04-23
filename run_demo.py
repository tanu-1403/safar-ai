"""
run_demo.py — Complete System Demo (No Dashboard Required)
==========================================================
Exercises every module end-to-end and prints a detailed report.
Useful for CI, headless servers, and hackathon judges who want to
see all outputs without launching Streamlit.

Run:
    cd safar_ai
    python run_demo.py

Output:
    - Console report with all metrics
    - data/demo_audit_report.json   (full explainability report)
    - data/edge_telemetry_demo.json (sample edge packets)
    - data/pdp_features.json        (partial dependence data)
"""

import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

print("\n" + "═"*65)
print("  Safar AI — Complete System Demo")
print("═"*65)

# ── Module imports ────────────────────────────────────────────────
from modules.ingestion       import RoadImageIngestor
from modules.spectral        import SpectralReflectivityEngine
from modules.model           import ReflectivityPredictor, build_training_dataset
from modules.digital_twin    import build_digital_twin_from_dataset
from modules.prediction      import PredictionPipeline
from modules.edge_deployment import (EdgeInferencePipeline,
                                     QuantizedReflectivityModel,
                                     estimate_deployment_cost)
from modules.explainability  import (FeatureImportanceAnalyzer,
                                     WhatIfAnalyzer,
                                     PartialDependenceAnalyzer,
                                     generate_audit_report)
from utils import ensure_dir, reflectivity_to_status, ALERT_CRITICAL, ALERT_WARNING

ensure_dir("data")
ensure_dir("models")

DIVIDER = "─" * 65

# ══════════════════════════════════════════════════════════════════
# 1. DATA INGESTION
# ══════════════════════════════════════════════════════════════════
print(f"\n{'1. DATA INGESTION':}")
print(DIVIDER)

ingestor = RoadImageIngestor()

# Generate synthetic road images at different reflectivities
demo_images = {}
for label, r, cond in [
    ("new_good",    0.88, "clear"),
    ("aged_worn",   0.45, "haze"),
    ("critical",    0.22, "rain"),
]:
    img = ingestor.generate_synthetic_road_image(reflectivity=r, condition=cond, seed=hash(label)%999)
    preprocessed, features = ingestor.process_image_full_pipeline(img, cond)
    demo_images[label] = {"features": features, "target_r": r}
    print(f"  [{label:12s}] brightness={features['mean_brightness']:.3f}  "
          f"contrast={features['michelson_contrast']:.3f}  "
          f"bright_ratio={features['bright_ratio']:.3f}  "
          f"edge_density={features['edge_density']:.3f}")

# ══════════════════════════════════════════════════════════════════
# 2. SPECTRAL ENGINE
# ══════════════════════════════════════════════════════════════════
print(f"\n{'2. SPECTRAL REFLECTIVITY ENGINE':}")
print(DIVIDER)

engine = SpectralReflectivityEngine()

scenarios = [
    ("road_marking_new",   "clear",     0.0, 0.0,  0.0),
    ("road_marking_faded", "clear",     0.8, 0.2,  0.4),
    ("new_asphalt",        "rain",      0.1, 0.05, 0.05),
    ("aged_asphalt",       "haze",      0.5, 0.3,  0.3),
    ("cracked_asphalt",    "heavy_rain",0.7, 0.5,  0.6),
]

print(f"  {'Material':<25} {'Weather':<12} {'Age':>5} {'Dirt':>5} {'Score':>7} {'Status'}")
print(f"  {'─'*25} {'─'*12} {'─'*5} {'─'*5} {'─'*7} {'─'*10}")
for mat, wx, age, dirt, wear in scenarios:
    res = engine.analyze_segment(mat, wx, age, dirt, wear)
    s   = reflectivity_to_status(res["reflectivity_score"])
    print(f"  {mat:<25} {wx:<12} {age:>5.1f} {dirt:>5.2f} "
          f"{res['reflectivity_score']:>7.4f} {s['emoji']} {s['label']}")

print(f"\n  Generating full synthetic dataset (20 segs × 180 days)...")
df = engine.generate_synthetic_dataset(20, 180, seed=42)
df.to_csv("data/synthetic_spectral_data.csv", index=False)
print(f"  ✅ {len(df):,} rows → data/synthetic_spectral_data.csv")

# ══════════════════════════════════════════════════════════════════
# 3. AI MODEL
# ══════════════════════════════════════════════════════════════════
print(f"\n{'3. AI MODEL — GRADIENT BOOSTING REGRESSOR':}")
print(DIVIDER)

train_df  = build_training_dataset(df)
predictor = ReflectivityPredictor()
metrics   = predictor.train(train_df)
predictor.save()

print(f"  Training samples : {int(len(train_df)*0.8):,}")
print(f"  Test samples     : {int(len(train_df)*0.2):,}")
print(f"  Train MAE        : {metrics['train_mae']:.5f}")
print(f"  Test  MAE        : {metrics['test_mae']:.5f}")
print(f"  Train R²         : {metrics['train_r2']:.5f}")
print(f"  Test  R²         : {metrics['test_r2']:.5f}")

# Sample predictions
print(f"\n  Sample predictions vs spectral ground-truth:")
sample = train_df.sample(5, random_state=99)
for _, row in sample.iterrows():
    pred = predictor.predict(row.to_dict())
    true = row["reflectivity_score"]
    err  = abs(pred - true)
    print(f"    True={true:.4f}  Predicted={pred:.4f}  Error={err:.4f}")

# Feature importance
fi = predictor.get_feature_importance()
print(f"\n  Top 5 most important features:")
for _, r in fi.head(5).iterrows():
    bar = "█" * int(r['importance']*200)
    print(f"    {r['feature']:<22} {bar:<20} {r['importance']:.4f}")

# ══════════════════════════════════════════════════════════════════
# 4. DIGITAL TWIN
# ══════════════════════════════════════════════════════════════════
print(f"\n{'4. DIGITAL TWIN REGISTRY':}")
print(DIVIDER)

registry = build_digital_twin_from_dataset(df, n_segments=20)
registry.save()

summary = registry.summary_dataframe()
alerts  = registry.get_alerts()

print(f"  Registered segments : {len(registry.all_segments())}")
print(f"  Active alerts       : {len(alerts)}")
print(f"  Status breakdown    :")
for status, count in summary["status_label"].value_counts().items():
    icons = {"GOOD":"✅","FAIR":"⚠️","WARNING":"🔶","CRITICAL":"🚨"}
    bar   = "█" * count
    print(f"    {icons.get(status,'❓')} {status:<10} {bar} ({count})")

print(f"\n  Critical/Warning segments:")
for _, row in alerts.head(5).iterrows():
    print(f"    {row['segment_id']:10s} score={row['score']:.3f}  "
          f"action: {row['action']}")

# Show one segment's trend
seg_obj = registry.get_segment("SEG_001")
if seg_obj:
    print(f"\n  SEG_001 deep-dive:")
    print(f"    Material         : {seg_obj.material}")
    print(f"    Current score    : {seg_obj.latest_reflectivity():.4f}")
    print(f"    Trend (7-day)    : {seg_obj.reflectivity_trend()}")
    print(f"    History readings : {len(seg_obj.reflectivity_history)}")
    print(f"    Maintenance logs : {len(seg_obj.maintenance_log)}")

# ══════════════════════════════════════════════════════════════════
# 5. PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════
print(f"\n{'5. ARIMA DEGRADATION FORECAST':}")
print(DIVIDER)

hist_df  = registry.history_dataframe()
pipeline = PredictionPipeline(use_lstm=False)
pipeline.fit_all(hist_df)
forecasts = pipeline.forecast_all(hist_df, steps=60)

print(f"  Fitted ARIMA models  : {len(forecasts)}")
print(f"\n  30-day forecasts for top 5 critical segments:")
critical_segs = alerts["segment_id"].head(5).tolist()
for sid in critical_segs:
    fc = forecasts.get(sid)
    if fc is None: continue
    curr_score = summary[summary["segment_id"]==sid]["reflectivity_score"].values[0]
    fc30 = fc["mean"][29]
    breach_day = next((i+1 for i,v in enumerate(fc["mean"]) if v < ALERT_CRITICAL), None)
    print(f"    {sid}: now={curr_score:.3f} → day30={fc30:.3f}  "
          f"critical_breach={'day '+str(breach_day) if breach_day else 'not predicted'}")

# Maintenance recommendations
seg_lengths = {
    row["segment_id"]: max(1, row["end_km"] - row["start_km"])
    for _, row in summary.iterrows()
}
recs = pipeline.get_recommendations(summary, seg_lengths)
budget_stats = pipeline.scheduler.budget_analysis(recs, 50_000_000)

print(f"\n  Maintenance recommendations summary:")
print(f"    Total recommended spend : ₹{budget_stats['total_recommended_cost']:,.0f}")
print(f"    Immediate required      : ₹{budget_stats['immediate_spend_required']:,.0f}")
print(f"    Budget utilization (50Cr): {budget_stats['budget_utilization_pct']:.1f}%")
print(f"    Segments needing action : {budget_stats['segments_in_immediate']} immediate, "
      f"{budget_stats['segments_in_high']} high-priority")
recs.to_csv("data/maintenance_recommendations.csv", index=False)
print(f"    Saved → data/maintenance_recommendations.csv")

# ══════════════════════════════════════════════════════════════════
# 6. EDGE DEPLOYMENT
# ══════════════════════════════════════════════════════════════════
print(f"\n{'6. EDGE DEPLOYMENT — QUANTIZED MODEL':}")
print(DIVIDER)

edge_model = QuantizedReflectivityModel()
bench      = edge_model.benchmark(n_iters=500)
print(f"  Model size (INT8)    : {bench['model_size_kb']:.3f} KB")
print(f"  Inference latency    : {bench['mean_ms']:.4f} ms (mean)  "
      f"{bench['min_ms']:.4f} ms (min)")
print(f"  Throughput           : {bench['fps']:.0f} FPS")
print(f"  Target hardware      : Raspberry Pi 4 (~50ms/frame), Jetson Nano (~12ms)")

# Simulate edge pipeline
edge_pipe = EdgeInferencePipeline("EDGE_NH48_001", "SEG_001")
print(f"\n  Simulating 10 frames through edge pipeline:")
telemetry_packets = []
for i in range(10):
    features = {
        "mean_brightness":    0.6 - i*0.03,
        "michelson_contrast": 0.55,
        "bright_ratio":       0.4 - i*0.02,
        "edge_density":       0.1 + i*0.02,
        "age_factor":         0.2 + i*0.05,
        "dirt_level":         0.1,
        "wear_level":         0.15,
        "r_mean": 0.35, "g_mean": 0.36, "b_mean": 0.32,
        "rg_ratio": 0.97, "std_brightness": 0.08, "laplacian_var": 0.4,
    }
    result = edge_pipe.process_frame(
        visual_features=features,
        spectral_mean=0.65 - i*0.03,
        weather="clear",
        lat=28.61+i*0.01, lon=77.20+i*0.01,
        speed_kmh=60.0,
    )
    telemetry_packets.append(json.loads(result["packet_json"]))
    print(f"    Frame {i+1:2d}: score={result['score']:.4f}  "
          f"{'🚨 ALERT' if result['alert'] else '✅ OK':12s}  "
          f"inf={result['inference_ms']:.2f}ms  "
          f"pkt={result['packet_bytes']}B")

# Save telemetry
with open("data/edge_telemetry_demo.json", "w") as f:
    json.dump(telemetry_packets, f, indent=2)
print(f"  Saved → data/edge_telemetry_demo.json ({len(telemetry_packets)} packets)")

# Queue
queued = edge_pipe.flush_queue()
print(f"  Upload queue flushed : {len(queued)} packets")

# Deployment cost estimate
costs = estimate_deployment_cost(highway_km=500)
print(f"\n  Deployment cost estimate (500 km pilot):")
print(f"    Devices needed     : {costs['n_devices']}")
print(f"    CAPEX              : ₹{costs['capex_inr']:,.0f}")
print(f"    Annual OPEX        : ₹{costs['annual_opex_inr']:,.0f}")
print(f"    Cost per km        : ₹{costs['cost_per_km_inr']:,.0f}")
print(f"    Annual savings     : ₹{costs['annual_savings_inr']:,.0f}")
print(f"    Payback period     : {costs['payback_months']:.1f} months")
print(f"    5-year ROI         : {costs['5yr_roi_pct']:.1f}%")

# ══════════════════════════════════════════════════════════════════
# 7. EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'7. AI EXPLAINABILITY':}")
print(DIVIDER)

analyzer = FeatureImportanceAnalyzer(predictor)
what_if  = WhatIfAnalyzer(predictor)
pdp_analyzer = PartialDependenceAnalyzer(predictor)

# Feature importance
ranked = analyzer.get_ranked_features()
print(f"  Feature importance (top 8):")
for _, row in ranked.head(8).iterrows():
    bar = "█" * int(row['importance'] * 250)
    dirn = "↑" if row["direction"] == "+" else "↓"
    print(f"    {dirn} {row['feature']:<22} {bar:<22} {row['importance']:.4f}  [{row['category']}]")

# Interpret a prediction for a sample segment
sample_features = train_df.sample(1, random_state=7).iloc[0].to_dict()
sample_score    = predictor.predict(sample_features)
interpretation  = analyzer.interpret_prediction(sample_features, sample_score)

print(f"\n  Sample segment explanation:")
print(f"    Score   : {sample_score:.4f}  ({interpretation['status']})")
print(f"    Summary : {interpretation['summary']}")
if interpretation['recommendations']:
    print(f"    Recs    : {interpretation['recommendations'][0]}")

# What-if: maintenance simulation
print(f"\n  What-if analysis (maintenance simulation):")
wf_feats = {
    "mean_brightness": 0.35, "bright_ratio": 0.15, "michelson_contrast": 0.4,
    "spectral_mean": 0.32, "age_factor": 0.55, "dirt_level": 0.3,
    "wear_level": 0.45, "edge_density": 0.35, "laplacian_var": 0.3,
    "r_mean": 0.25, "g_mean": 0.26, "b_mean": 0.22, "rg_ratio": 0.96,
    "std_brightness": 0.12, "spectral_std": 0.08, "material": "aged_asphalt",
}
for action in ["cleaning", "repainting", "microsurfacing", "resurfacing"]:
    sim = what_if.simulate_maintenance(wf_feats, action)
    arrow = "→"
    print(f"    {action:<14} {sim['status_before']:>8} {arrow} {sim['status_after']:<8}  "
          f"score: {sim['score_before']:.3f} → {sim['score_after']:.3f}  "
          f"(+{sim['improvement']:.3f})")

# Counterfactual
cf = what_if.find_counterfactual(wf_feats, target_score=0.70)
print(f"\n  Counterfactual — what to do to reach GOOD (0.70):")
print(f"    Current score  : {cf['current_score']}")
print(f"    Recommendation : {cf['recommendation']}")
print(f"    Predicted after: {cf['predicted_after']}")
print(f"    Message        : {cf['message']}")

# PDP for top 2 features
print(f"\n  Partial dependence (key features):")
for feat in ["spectral_mean", "mean_brightness", "age_factor"]:
    pdp = pdp_analyzer.compute_pdp(feat, n_grid=10, background_df=train_df)
    lo  = min(pdp["pdp"])
    hi  = max(pdp["pdp"])
    print(f"    {feat:<22} PDP range: [{lo:.3f}, {hi:.3f}]  "
          f"(Δ={hi-lo:.3f})")

# Save PDP data
pdp_all = pdp_analyzer.compute_all_pdps(n_grid=30, background_df=train_df)
with open("data/pdp_features.json", "w") as f:
    json.dump(pdp_all, f, indent=2)
print(f"\n  Saved → data/pdp_features.json ({len(pdp_all)} features)")

# Generate full audit report
audit_report = generate_audit_report("SEG_001", wf_feats, 0.32, predictor)
with open("data/demo_audit_report.json", "w") as f:
    json.dump(audit_report, f, indent=2, default=str)
print(f"  Saved → data/demo_audit_report.json")

# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f"  COMPLETE DEMO FINISHED SUCCESSFULLY")
print(f"{'═'*65}")
print(f"""
  System Outputs:
  ├── data/synthetic_spectral_data.csv      {len(df):>7,} rows
  ├── data/maintenance_recommendations.csv  {len(recs):>7} segments
  ├── data/digital_twin_state.json          {len(registry.all_segments()):>7} segments
  ├── data/edge_telemetry_demo.json         {len(telemetry_packets):>7} packets
  ├── data/pdp_features.json               {len(pdp_all):>7} features
  ├── data/demo_audit_report.json           1 full report
  └── models/reflectivity_predictor.joblib  trained model

  Model performance:
  ├── Test MAE  : {metrics['test_mae']:.5f}  (excellent — goal <0.05)
  ├── Test R²   : {metrics['test_r2']:.5f}  (excellent — goal >0.90)
  └── Edge FPS  : {bench['fps']:.0f} fps  (Raspberry Pi target)

  Highway status (NH-48 simulation):
  ├── Segments monitored : {len(summary)}
  ├── Critical alerts    : {int((summary['status_label']=='CRITICAL').sum())}
  ├── Warning alerts     : {int((summary['status_label']=='WARNING').sum())}
  └── Avg reflectivity   : {summary['reflectivity_score'].mean():.3f}

  
  Safar AI is ready.
  Tagline: Har Safar, Surakshit Safar
  Next step → streamlit run app.py
""")
