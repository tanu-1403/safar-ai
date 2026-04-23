"""
modules/model.py — AI Model Module
====================================
Provides two complementary models:
  1. CNN Feature Extractor — extracts deep visual embeddings from road images
     (MobileNet-style lightweight architecture, TensorFlow/Keras)
  2. Gradient Boosting Regressor — predicts reflectivity score from
     combined spectral + visual + CNN features
  3. SHAP explainability for the regression model

Author: Safar AI Team
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from typing import Dict, List, Optional, Tuple

# Soft-import TensorFlow (graceful fallback if not installed)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Soft-import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from utils import ensure_dir

logger = logging.getLogger("SafarAI.model")

MODELS_DIR = ensure_dir("models")

# Feature columns used by the regression model
FEATURE_COLS = [
    "mean_brightness", "std_brightness", "michelson_contrast",
    "edge_density", "laplacian_var", "bright_ratio",
    "r_mean", "g_mean", "b_mean", "rg_ratio",
    "spectral_mean", "spectral_std",
    "age_factor", "dirt_level", "wear_level",
]


# ─────────────────────────────────────────────
# 1. Lightweight CNN Feature Extractor
# ─────────────────────────────────────────────

def build_cnn_feature_extractor(input_shape: Tuple = (224, 224, 3)) -> Optional[object]:
    """
    Build a lightweight MobileNet-inspired CNN feature extractor.
    Returns a Keras Model that maps (224,224,3) → (128,) feature vector.

    Falls back to None if TensorFlow is not available.

    Architecture:
      Conv→BN→ReLU blocks (depthwise separable) → GlobalAveragePooling → Dense(128)
    """
    if not TF_AVAILABLE:
        logger.warning("TensorFlow not available — CNN extractor disabled.")
        return None

    def dw_block(x, filters, stride=1):
        """Depthwise separable convolution block (MobileNet-style)."""
        x = layers.DepthwiseConv2D(3, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    inp = layers.Input(shape=input_shape, name="road_image")

    # Initial standard conv
    x = layers.Conv2D(32, 3, strides=2, padding='same', name="stem_conv")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise separable blocks (lightweight)
    x = dw_block(x, 64)
    x = dw_block(x, 128, stride=2)
    x = dw_block(x, 128)
    x = dw_block(x, 256, stride=2)
    x = dw_block(x, 256)
    x = dw_block(x, 512, stride=2)

    # Global pooling + embedding
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation='relu', name="embedding")(x)
    x = layers.Dropout(0.3)(x)

    model = Model(inputs=inp, outputs=x, name="RoadCNNFeatureExtractor")
    logger.info("CNN feature extractor built | params=%d",
                model.count_params())
    return model


class CNNExtractor:
    """
    Wrapper around the CNN feature extractor.
    Can extract 128-dim embeddings from road images.
    """

    def __init__(self):
        self.model = build_cnn_feature_extractor()
        self._compiled = False

    def extract(self, images: np.ndarray) -> np.ndarray:
        """
        Extract feature embeddings from a batch of images.

        Args:
            images: float32 array of shape (N, 224, 224, 3) in [0, 1].

        Returns:
            float32 array of shape (N, 128) or zeros if TF unavailable.
        """
        if self.model is None:
            logger.warning("CNN unavailable — returning zero embeddings.")
            return np.zeros((len(images), 128), dtype=np.float32)

        if images.ndim == 3:
            images = np.expand_dims(images, 0)

        embeddings = self.model.predict(images, verbose=0)
        return embeddings


# ─────────────────────────────────────────────
# 2. Gradient Boosting Reflectivity Regressor
# ─────────────────────────────────────────────

class ReflectivityPredictor:
    """
    Gradient Boosting model to predict reflectivity score from
    a combined feature vector (visual + spectral + environmental).

    Features used (15):
      - Visual: mean_brightness, std_brightness, michelson_contrast,
                edge_density, laplacian_var, bright_ratio,
                r_mean, g_mean, b_mean, rg_ratio
      - Spectral: spectral_mean, spectral_std
      - Environmental: age_factor, dirt_level, wear_level

    Target: reflectivity_score ∈ [0, 1]
    """

    def __init__(self):
        self.model   = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        self.scaler  = StandardScaler()
        self.trained = False
        self.feature_names = FEATURE_COLS
        self.metrics = {}

    def _make_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and align feature matrix from a DataFrame."""
        available = [c for c in FEATURE_COLS if c in df.columns]
        missing   = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            logger.warning("Missing feature columns (will use zeros): %s", missing)
        X = df.reindex(columns=FEATURE_COLS, fill_value=0.0)[FEATURE_COLS].values
        return X.astype(np.float32)

    def train(self, df: pd.DataFrame, target_col: str = "reflectivity_score") -> Dict:
        """
        Train the model on a dataset DataFrame.

        Args:
            df: DataFrame with feature columns + target column.
            target_col: Name of the target column.

        Returns:
            Dict with train/test MAE and R² metrics.
        """
        X = self._make_features(df)
        y = df[target_col].values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        logger.info("Training ReflectivityPredictor on %d samples...", len(X_train))
        self.model.fit(X_train_s, y_train)
        self.trained = True

        train_pred = self.model.predict(X_train_s)
        test_pred  = self.model.predict(X_test_s)

        self.metrics = {
            "train_mae": round(mean_absolute_error(y_train, train_pred), 4),
            "test_mae":  round(mean_absolute_error(y_test,  test_pred),  4),
            "train_r2":  round(r2_score(y_train, train_pred), 4),
            "test_r2":   round(r2_score(y_test,  test_pred),  4),
        }
        logger.info("Training complete | %s", self.metrics)
        return self.metrics

    def predict(self, features: Dict) -> float:
        """
        Predict reflectivity score for a single sample.

        Args:
            features: Dict with feature names → values.

        Returns:
            Predicted reflectivity score in [0, 1].
        """
        if not self.trained:
            logger.warning("Model not trained — returning spectral_mean as fallback.")
            return float(features.get("spectral_mean", 0.5))

        row = np.array([[features.get(f, 0.0) for f in FEATURE_COLS]], dtype=np.float32)
        row_s = self.scaler.transform(row)
        pred  = self.model.predict(row_s)[0]
        return float(np.clip(pred, 0.0, 1.0))

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Predict reflectivity scores for a full DataFrame."""
        if not self.trained:
            return df.get("spectral_mean", pd.Series(np.zeros(len(df)))).values
        X = self._make_features(df)
        X_s = self.scaler.transform(X)
        return np.clip(self.model.predict(X_s), 0.0, 1.0)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance scores from the gradient boosting model.

        Returns:
            DataFrame with columns ['feature', 'importance'] sorted descending.
        """
        if not self.trained:
            return pd.DataFrame({"feature": FEATURE_COLS, "importance": [0.0]*len(FEATURE_COLS)})

        imp = self.model.feature_importances_
        df  = pd.DataFrame({
            "feature":    self.feature_names,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    def explain_with_shap(self, df: pd.DataFrame, max_samples: int = 100) -> Optional[object]:
        """
        Compute SHAP values for explainability.

        Args:
            df: Input DataFrame with feature columns.
            max_samples: Limit samples for performance.

        Returns:
            SHAP Explanation object or None if SHAP unavailable.
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available — install with: pip install shap")
            return None
        if not self.trained:
            logger.warning("Model not trained — cannot compute SHAP values.")
            return None

        X      = self._make_features(df.head(max_samples))
        X_s    = self.scaler.transform(X)
        X_df   = pd.DataFrame(X_s, columns=self.feature_names)

        explainer = shap.TreeExplainer(self.model)
        shap_vals = explainer(X_df)
        return shap_vals

    def save(self, path: str = None):
        """Persist model + scaler to disk."""
        path = path or os.path.join(MODELS_DIR, "reflectivity_predictor.joblib")
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "metrics": self.metrics, "trained": self.trained}, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str = None):
        """Load model + scaler from disk."""
        path = path or os.path.join(MODELS_DIR, "reflectivity_predictor.joblib")
        if not os.path.exists(path):
            logger.warning("No saved model at %s", path)
            return
        data = joblib.load(path)
        self.model   = data["model"]
        self.scaler  = data["scaler"]
        self.metrics = data["metrics"]
        self.trained = data["trained"]
        logger.info("Model loaded from %s | metrics=%s", path, self.metrics)


# ─────────────────────────────────────────────
# 3. Synthetic Feature Dataset Builder
# ─────────────────────────────────────────────

def build_training_dataset(spectral_df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment the spectral time-series DataFrame with synthetic visual features,
    creating a complete training dataset for the regressor.

    Adds 10 visual feature columns derived from reflectivity + noise
    (simulating what real cameras would capture).

    Args:
        spectral_df: DataFrame from SpectralReflectivityEngine.generate_synthetic_dataset()

    Returns:
        DataFrame with all FEATURE_COLS + target 'reflectivity_score'.
    """
    np.random.seed(0)
    df = spectral_df.copy()
    n  = len(df)
    r  = df["reflectivity_score"].values

    # Simulate visual features correlated with reflectivity + noise
    df["mean_brightness"]    = np.clip(r * 0.85 + np.random.normal(0, 0.04, n), 0, 1)
    df["std_brightness"]     = np.clip(0.05 + (1-r)*0.12 + np.random.normal(0, 0.01, n), 0, 0.3)
    df["michelson_contrast"] = np.clip(r * 0.75 + np.random.normal(0, 0.05, n), 0, 1)
    df["edge_density"]       = np.clip((1-r)*0.25 + np.random.normal(0, 0.02, n), 0, 1)
    df["laplacian_var"]      = np.clip(r * 0.6 + np.random.normal(0, 0.05, n), 0, 1)
    df["bright_ratio"]       = np.clip(r * 0.70 + np.random.normal(0, 0.04, n), 0, 1)
    df["r_mean"]             = np.clip(r * 0.40 + np.random.normal(0, 0.03, n), 0, 1)
    df["g_mean"]             = np.clip(r * 0.42 + np.random.normal(0, 0.03, n), 0, 1)
    df["b_mean"]             = np.clip(r * 0.38 + np.random.normal(0, 0.03, n), 0, 1)
    df["rg_ratio"]           = np.clip(df["r_mean"] / (df["g_mean"] + 1e-6), 0.5, 2.0)

    # Ensure spectral_std column exists
    if "spectral_std" not in df.columns:
        df["spectral_std"] = np.clip(0.05 + (1-r)*0.08 + np.random.normal(0, 0.01, n), 0, 0.3)

    logger.info("Training dataset built: %d rows × %d features", len(df), len(FEATURE_COLS))
    return df
