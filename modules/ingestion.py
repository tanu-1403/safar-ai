"""
modules/ingestion.py — Data Ingestion Module
=============================================
Handles road image/video input processing.
- Loads images from disk or synthetic generation
- Applies preprocessing (resize, normalize, edge enhancement)
- Extracts per-pixel brightness, contrast, and texture features
- Returns feature vectors ready for the Spectral Engine and AI Model

Author: Safar AI Team
"""

import cv2
import numpy as np
import os
import logging
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger("SafarAI.ingestion")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
TARGET_SIZE   = (224, 224)   # Standard input size (matches MobileNet / ResNet)
ROAD_REGION_Y = 0.40         # Only analyze lower 60% of image (road region)


# ─────────────────────────────────────────────
# Core Ingestion Class
# ─────────────────────────────────────────────

class RoadImageIngestor:
    """
    Ingests road images/frames and extracts visual features for
    downstream spectral and AI processing.

    Attributes:
        target_size: (width, height) to resize all images.
        road_fraction: Fraction of image height to crop from top (sky removal).
    """

    def __init__(self, target_size: Tuple[int,int] = TARGET_SIZE,
                 road_fraction: float = ROAD_REGION_Y):
        self.target_size    = target_size
        self.road_fraction  = road_fraction
        logger.info("RoadImageIngestor initialized | target_size=%s", target_size)

    # ── Image Loading ──────────────────────────────────────────────────

    def load_image(self, path: str) -> Optional[np.ndarray]:
        """
        Load an image from disk in BGR format.

        Args:
            path: File path to image.

        Returns:
            BGR numpy array or None on failure.
        """
        if not os.path.exists(path):
            logger.warning("Image not found: %s", path)
            return None
        img = cv2.imread(path)
        if img is None:
            logger.error("OpenCV failed to read: %s", path)
        return img

    def load_video_frames(self, path: str, max_frames: int = 30) -> List[np.ndarray]:
        """
        Extract evenly-spaced frames from a video file.

        Args:
            path: Video file path.
            max_frames: Maximum number of frames to extract.

        Returns:
            List of BGR numpy arrays.
        """
        cap    = cv2.VideoCapture(path)
        frames = []
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step   = max(1, total // max_frames)

        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            if len(frames) >= max_frames:
                break

        cap.release()
        logger.info("Extracted %d frames from %s", len(frames), path)
        return frames

    # ── Synthetic Image Generation ─────────────────────────────────────

    def generate_synthetic_road_image(
        self,
        reflectivity: float = 0.7,
        condition: str = "clear",
        seed: int = 42
    ) -> np.ndarray:
        """
        Generate a synthetic road image with controllable reflectivity.
        Used when real camera data is unavailable (simulation mode).

        Args:
            reflectivity: Target reflectivity (0–1). Higher → brighter road markings.
            condition: Weather condition string ('clear', 'rain', 'fog', 'haze').
            seed: Random seed for reproducibility.

        Returns:
            BGR numpy array of shape (224, 224, 3).
        """
        np.random.seed(seed)
        h, w = self.target_size[1], self.target_size[0]

        # ── Base asphalt layer ────────────────────────────────────────
        base_brightness = int(reflectivity * 80 + 30)
        road = np.full((h, w, 3), base_brightness, dtype=np.uint8)

        # Add asphalt texture (granular noise)
        texture = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        road    = np.clip(road.astype(np.int16) + texture, 0, 255).astype(np.uint8)

        # ── Road markings ─────────────────────────────────────────────
        marking_brightness = int(reflectivity * 200 + 55)

        # Center dashed yellow line
        for y in range(0, h, 30):
            cv2.line(road, (w//2, y), (w//2, y+18),
                     (0, marking_brightness, marking_brightness), 3)

        # Left & right edge lines
        cv2.line(road, (w//5, 0),     (w//5, h),     (marking_brightness,)*3, 2)
        cv2.line(road, (4*w//5, 0),   (4*w//5, h),   (marking_brightness,)*3, 2)

        # Lane dividers
        for y in range(0, h, 40):
            cv2.line(road, (2*w//5, y), (2*w//5, y+20),
                     (marking_brightness,)*3, 2)
            cv2.line(road, (3*w//5, y), (3*w//5, y+20),
                     (marking_brightness,)*3, 2)

        # ── Wear/crack simulation for low reflectivity ─────────────────
        if reflectivity < 0.5:
            n_cracks = int((0.5 - reflectivity) * 40)
            for _ in range(n_cracks):
                x1 = np.random.randint(0, w)
                y1 = np.random.randint(0, h)
                x2 = x1 + np.random.randint(-30, 30)
                y2 = y1 + np.random.randint(10, 40)
                cv2.line(road, (x1, y1), (x2, y2), (20, 20, 20), 1)

        # ── Weather overlay ───────────────────────────────────────────
        road = self._apply_weather(road, condition)

        return road

    def _apply_weather(self, img: np.ndarray, condition: str) -> np.ndarray:
        """Apply a weather-based overlay to a road image."""
        if condition == "fog":
            fog_layer = np.full_like(img, 200)
            img = cv2.addWeighted(img, 0.55, fog_layer, 0.45, 0)

        elif condition in ("rain", "heavy_rain"):
            intensity = 0.7 if condition == "heavy_rain" else 0.85
            rain_layer = np.zeros_like(img)
            n_drops = 300 if condition == "heavy_rain" else 150
            for _ in range(n_drops):
                x  = np.random.randint(0, img.shape[1])
                y  = np.random.randint(0, img.shape[0])
                x2 = x + np.random.randint(-3, 3)
                y2 = y + np.random.randint(5, 15)
                cv2.line(rain_layer, (x, y), (x2, y2), (180, 180, 220), 1)
            img = cv2.addWeighted(img, intensity, rain_layer, 1 - intensity, 0)

        elif condition == "haze":
            haze_layer = np.full_like(img, 160)
            img = cv2.addWeighted(img, 0.75, haze_layer, 0.25, 0)

        return img

    # ── Preprocessing Pipeline ─────────────────────────────────────────

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline:
        1. Crop road region (remove sky)
        2. Resize to target size
        3. CLAHE contrast enhancement
        4. Normalize to [0, 1]

        Args:
            img: Raw BGR image.

        Returns:
            Float32 numpy array in [0, 1], shape = (H, W, 3).
        """
        # Crop to road region
        h = img.shape[0]
        crop_y = int(h * self.road_fraction)
        img = img[crop_y:, :]

        # Resize
        img = cv2.resize(img, self.target_size)

        # CLAHE on L channel (Lab color space) for contrast normalization
        lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Normalize to float32 [0, 1]
        return img.astype(np.float32) / 255.0

    # ── Feature Extraction ─────────────────────────────────────────────

    def extract_visual_features(self, img_preprocessed: np.ndarray) -> Dict[str, float]:
        """
        Extract numerical visual features from a preprocessed road image.
        These features feed the spectral engine and regression model.

        Args:
            img_preprocessed: Float32 image in [0, 1].

        Returns:
            Dictionary of named feature scalars.
        """
        gray = cv2.cvtColor(
            (img_preprocessed * 255).astype(np.uint8),
            cv2.COLOR_BGR2GRAY
        )

        # Brightness statistics
        mean_brightness = float(np.mean(img_preprocessed))
        std_brightness  = float(np.std(img_preprocessed))

        # Contrast (Michelson)
        pmin = float(np.percentile(gray, 2))
        pmax = float(np.percentile(gray, 98))
        michelson_contrast = (pmax - pmin) / (pmax + pmin + 1e-6)

        # Edge density (proxy for crack/wear detection)
        edges     = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))

        # Texture (Laplacian variance — sharpness proxy)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # High-reflectivity pixel ratio (bright markings)
        bright_ratio = float(np.mean(img_preprocessed > 0.75))

        # Color channel ratios (detect yellowing/fading)
        r_mean = float(np.mean(img_preprocessed[:, :, 2]))   # R channel (BGR → idx 2)
        g_mean = float(np.mean(img_preprocessed[:, :, 1]))
        b_mean = float(np.mean(img_preprocessed[:, :, 0]))
        rg_ratio = r_mean / (g_mean + 1e-6)

        return {
            "mean_brightness":    mean_brightness,
            "std_brightness":     std_brightness,
            "michelson_contrast": michelson_contrast,
            "edge_density":       edge_density,
            "laplacian_var":      laplacian_var / 10000.0,  # normalize
            "bright_ratio":       bright_ratio,
            "r_mean":             r_mean,
            "g_mean":             g_mean,
            "b_mean":             b_mean,
            "rg_ratio":           rg_ratio,
        }

    def process_image_full_pipeline(
        self,
        img_bgr: np.ndarray,
        condition: str = "clear"
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        End-to-end: raw BGR image → preprocessed array + feature dict.

        Args:
            img_bgr: Raw BGR image from camera.
            condition: Weather/environmental condition label.

        Returns:
            (preprocessed_image, feature_dict)
        """
        preprocessed = self.preprocess(img_bgr)
        features     = self.extract_visual_features(preprocessed)
        features["condition"] = condition
        return preprocessed, features
