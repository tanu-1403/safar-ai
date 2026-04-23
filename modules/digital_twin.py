"""
modules/digital_twin.py — Digital Twin Module
==============================================
Maintains a virtual, evolving representation of highway segments.

A Digital Twin in this context means:
  - Each physical highway segment has a virtual counterpart
  - The twin stores: geometry, location, material, sensor history,
    reflectivity time-series, alert states, and maintenance logs
  - The twin is updated in real-time (or in simulation) as new
    sensor readings arrive
  - Queries like "show all CRITICAL segments on NH-48" are answered
    by the twin without needing to re-process raw data

Data Model:
  HighwaySegment (dataclass)
    ├── id, name, highway_name
    ├── start_km, end_km, lat, lon
    ├── material, lane_count, surface_area_m2
    ├── reflectivity_history: List[ReflectivityReading]
    └── maintenance_log: List[MaintenanceEvent]

Author: Safar AI Team
"""

import numpy as np
import pandas as pd
import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

from utils import (
    reflectivity_to_status, ALERT_CRITICAL, ALERT_WARNING,
    HISTORY_DAYS, ensure_dir, timestamp_range
)

logger = logging.getLogger("SafarAI.digital_twin")


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class ReflectivityReading:
    """Single reflectivity observation for a segment."""
    timestamp:           str
    reflectivity_score:  float
    weather:             str  = "clear"
    spectral_mean:       float = 0.0
    age_factor:          float = 0.0
    dirt_level:          float = 0.0
    wear_level:          float = 0.0
    source:              str  = "sensor"  # 'sensor', 'camera', 'manual', 'predicted'


@dataclass
class MaintenanceEvent:
    """Record of a maintenance action on a segment."""
    event_date:      str
    event_type:      str   # 'repainting', 'resurfacing', 'cleaning', 'inspection'
    performed_by:    str   = "NHAI Field Team"
    notes:           str   = ""
    cost_estimate:   float = 0.0  # INR


@dataclass
class HighwaySegment:
    """
    Virtual representation of one highway segment.
    This is the core Digital Twin entity.
    """
    segment_id:         str
    highway_name:       str
    start_km:           float
    end_km:             float
    lat:                float
    lon:                float
    material:           str         = "new_asphalt"
    lane_count:         int         = 4
    surface_area_m2:    float       = 1000.0
    created_at:         str         = field(default_factory=lambda: datetime.now().isoformat())
    reflectivity_history: List[ReflectivityReading] = field(default_factory=list)
    maintenance_log:     List[MaintenanceEvent]      = field(default_factory=list)

    # ── Computed Properties ──────────────────────────────────────────

    def latest_reflectivity(self) -> Optional[float]:
        """Return the most recent reflectivity score, or None."""
        if not self.reflectivity_history:
            return None
        return self.reflectivity_history[-1].reflectivity_score

    def status(self) -> Dict:
        """Return current status dict based on latest reflectivity."""
        score = self.latest_reflectivity()
        if score is None:
            return {"label": "NO DATA", "color": "#94a3b8", "emoji": "❓", "action": "Awaiting sensor data"}
        return reflectivity_to_status(score)

    def reflectivity_trend(self, window: int = 7) -> str:
        """
        Compute recent trend: 'degrading', 'stable', or 'improving'.
        Uses the last `window` readings.
        """
        if len(self.reflectivity_history) < window + 1:
            return "insufficient_data"

        recent = [r.reflectivity_score for r in self.reflectivity_history[-window:]]
        slope  = np.polyfit(range(len(recent)), recent, 1)[0]

        if slope < -0.002:
            return "degrading"
        elif slope > 0.002:
            return "improving"
        return "stable"

    def days_since_maintenance(self) -> Optional[int]:
        """Return days since last maintenance event, or None."""
        if not self.maintenance_log:
            return None
        last_date = datetime.fromisoformat(self.maintenance_log[-1].event_date)
        return (datetime.now() - last_date).days

    def add_reading(self, reading: ReflectivityReading):
        """Append a new reflectivity reading."""
        self.reflectivity_history.append(reading)

    def add_maintenance(self, event: MaintenanceEvent):
        """Append a maintenance event and reset wear/dirt tracking."""
        self.maintenance_log.append(event)
        # Simulate post-maintenance reflectivity boost
        if self.reflectivity_history:
            last = self.reflectivity_history[-1]
            boosted = min(0.95, last.reflectivity_score + 0.35)
            self.add_reading(ReflectivityReading(
                timestamp=event.event_date,
                reflectivity_score=boosted,
                weather="clear",
                source="manual",
            ))

    def to_dict(self) -> Dict:
        """Serialize to plain dict (for JSON export)."""
        return asdict(self)

    def to_summary_row(self) -> Dict:
        """Return a flat summary row for DataFrame display."""
        score = self.latest_reflectivity()
        status = self.status()
        return {
            "segment_id":          self.segment_id,
            "highway":             self.highway_name,
            "start_km":            self.start_km,
            "end_km":              self.end_km,
            "lat":                 self.lat,
            "lon":                 self.lon,
            "material":            self.material,
            "reflectivity_score":  round(score, 3) if score else None,
            "status_label":        status["label"],
            "status_color":        status["color"],
            "status_emoji":        status["emoji"],
            "action":              status["action"],
            "trend":               self.reflectivity_trend(),
            "n_readings":          len(self.reflectivity_history),
            "days_since_maintenance": self.days_since_maintenance(),
        }


# ─────────────────────────────────────────────
# Digital Twin Registry
# ─────────────────────────────────────────────

class DigitalTwinRegistry:
    """
    Central registry of all highway segment digital twins.

    Acts as an in-memory database with persistence to disk.
    Provides query interface for dashboard and alert system.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir  = ensure_dir(data_dir)
        self.segments: Dict[str, HighwaySegment] = {}
        self._twin_path = os.path.join(data_dir, "digital_twin_state.json")
        logger.info("DigitalTwinRegistry initialized | data_dir=%s", data_dir)

    # ── Segment Management ─────────────────────────────────────────────

    def register_segment(self, seg: HighwaySegment):
        """Add or overwrite a segment in the registry."""
        self.segments[seg.segment_id] = seg

    def get_segment(self, segment_id: str) -> Optional[HighwaySegment]:
        """Retrieve a segment by ID."""
        return self.segments.get(segment_id)

    def all_segments(self) -> List[HighwaySegment]:
        """Return all registered segments."""
        return list(self.segments.values())

    def summary_dataframe(self) -> pd.DataFrame:
        """Build a summary DataFrame of current segment states."""
        rows = [seg.to_summary_row() for seg in self.segments.values()]
        return pd.DataFrame(rows)

    def history_dataframe(self, segment_id: str = None) -> pd.DataFrame:
        """
        Return full reflectivity history as a DataFrame.

        Args:
            segment_id: If provided, filter to one segment.
        """
        rows = []
        segs = [self.segments[segment_id]] if segment_id else self.all_segments()
        for seg in segs:
            for r in seg.reflectivity_history:
                rows.append({
                    "segment_id":        seg.segment_id,
                    "highway":           seg.highway_name,
                    "timestamp":         r.timestamp,
                    "reflectivity_score": r.reflectivity_score,
                    "weather":           r.weather,
                    "spectral_mean":     r.spectral_mean,
                    "age_factor":        r.age_factor,
                    "source":            r.source,
                })
        df = pd.DataFrame(rows)
        if len(df):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(["segment_id", "timestamp"])
        return df

    # ── Alert Engine ───────────────────────────────────────────────────

    def get_alerts(self) -> pd.DataFrame:
        """
        Return all segments that require attention, sorted by urgency.
        """
        rows = []
        for seg in self.all_segments():
            score = seg.latest_reflectivity()
            if score is None:
                continue
            if score < ALERT_WARNING:
                status = reflectivity_to_status(score)
                rows.append({
                    "segment_id":   seg.segment_id,
                    "highway":      seg.highway_name,
                    "score":        round(score, 3),
                    "status":       status["label"],
                    "action":       status["action"],
                    "trend":        seg.reflectivity_trend(),
                    "last_km":      f"{seg.start_km:.0f}–{seg.end_km:.0f}",
                    "lat":          seg.lat,
                    "lon":          seg.lon,
                })
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values("score", ascending=True)
        return df

    # ── Persistence ────────────────────────────────────────────────────

    def save(self):
        """Serialize registry to JSON file."""
        state = {sid: seg.to_dict() for sid, seg in self.segments.items()}
        with open(self._twin_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info("Digital twin state saved to %s", self._twin_path)

    def load(self):
        """Deserialize registry from JSON file."""
        if not os.path.exists(self._twin_path):
            logger.info("No saved twin state found at %s", self._twin_path)
            return
        with open(self._twin_path) as f:
            state = json.load(f)
        for sid, data in state.items():
            readings = [ReflectivityReading(**r) for r in data.pop("reflectivity_history", [])]
            maint    = [MaintenanceEvent(**m)    for m in data.pop("maintenance_log", [])]
            seg = HighwaySegment(**data)
            seg.reflectivity_history = readings
            seg.maintenance_log      = maint
            self.segments[sid] = seg
        logger.info("Loaded %d segments from saved state", len(self.segments))


# ─────────────────────────────────────────────
# Factory: Build Twin from Synthetic Dataset
# ─────────────────────────────────────────────

def build_digital_twin_from_dataset(
    df: pd.DataFrame,
    highway_name: str = "NH-48 (Delhi–Mumbai)",
    n_segments: int = 20,
) -> DigitalTwinRegistry:
    """
    Construct a fully-populated DigitalTwinRegistry from a synthetic DataFrame.

    Args:
        df: Output of SpectralReflectivityEngine.generate_synthetic_dataset()
        highway_name: Name of the highway corridor.
        n_segments: Number of segments to create.

    Returns:
        Populated DigitalTwinRegistry.
    """
    np.random.seed(42)
    registry = DigitalTwinRegistry()

    # Representative coordinates along NH-48 (Delhi → Mumbai)
    # (simplified linear interpolation between endpoints)
    delhi_lat, delhi_lon     = 28.6139, 77.2090
    mumbai_lat, mumbai_lon   = 19.0760, 72.8777

    materials = ["new_asphalt","aged_asphalt","road_marking_new",
                 "worn_asphalt","concrete","road_marking_faded"]

    for i in range(n_segments):
        seg_id    = f"SEG_{i+1:03d}"
        frac      = i / max(n_segments - 1, 1)
        lat       = delhi_lat  + (mumbai_lat  - delhi_lat)  * frac + np.random.normal(0, 0.05)
        lon       = delhi_lon  + (mumbai_lon  - delhi_lon)  * frac + np.random.normal(0, 0.05)
        start_km  = round(frac * 1400, 1)
        end_km    = round(start_km + 70, 1)
        material  = materials[i % len(materials)]

        seg = HighwaySegment(
            segment_id=seg_id,
            highway_name=highway_name,
            start_km=start_km,
            end_km=end_km,
            lat=round(lat, 4),
            lon=round(lon, 4),
            material=material,
            lane_count=4 if i < 15 else 6,
            surface_area_m2=round((end_km - start_km) * 1000 * 7.5, 0),
        )

        # Populate history from DataFrame
        seg_df = df[df["segment_id"] == seg_id].copy()
        if seg_df.empty:
            # Use first available segment data if ID not found
            seg_df = df[df["segment_id"] == df["segment_id"].unique()[i % len(df["segment_id"].unique())]].copy()

        for _, row in seg_df.iterrows():
            reading = ReflectivityReading(
                timestamp=str(row["date"]),
                reflectivity_score=float(row["reflectivity_score"]),
                weather=str(row.get("weather", "clear")),
                spectral_mean=float(row.get("spectral_mean", 0.0)),
                age_factor=float(row.get("age_factor", 0.0)),
                dirt_level=float(row.get("dirt_level", 0.0)),
                wear_level=float(row.get("wear_level", 0.0)),
                source="synthetic_sensor",
            )
            seg.add_reading(reading)

        # Add simulated past maintenance for some segments
        if np.random.rand() < 0.4:  # 40% of segments had recent maintenance
            maint_days_ago = np.random.randint(30, 120)
            maint_date = (datetime.now() - timedelta(days=maint_days_ago)).isoformat()
            seg.maintenance_log.append(MaintenanceEvent(
                event_date=maint_date,
                event_type=np.random.choice(["repainting", "resurfacing", "cleaning"]),
                cost_estimate=round(np.random.uniform(50000, 500000), 0),
            ))

        registry.register_segment(seg)

    logger.info("Digital twin built: %d segments | highway=%s", n_segments, highway_name)
    return registry
