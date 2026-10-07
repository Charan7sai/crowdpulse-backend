# ──────────────────────────────────────────────
# Crowd Risk Monitoring — Configuration
# ──────────────────────────────────────────────
# Values are read from my.env via python-dotenv.
#
# IMPORTANT — capacity design:
#   ZONE_MAX_CAPACITY and ZONE_SAFE_CAPACITY start as None.
#   They are set at runtime by zone_estimator.py using
#   YOLOv8-seg floor detection + Fruin safety standards.
#   Do NOT set them manually — let the AI infer them.
#   ZONE_AREA_M2 is only used as a last-resort fallback
#   if the camera fails or the first frame is too dark.

import os

# ── Zone identity ──────────────────────────────────────────
ZONE_NAME = os.getenv("ZONE_NAME", "Main Hall")

# ── Zone capacity — set by AI at runtime ──────────────────
# These are intentionally None at startup.
# zone_estimator.calibrate() sets them from the camera feed.
ZONE_MAX_CAPACITY  = None   # absolute physical max (AI-inferred)
ZONE_SAFE_CAPACITY = None   # operational safe limit (AI-inferred)

# ── Fallback area — used ONLY if AI calibration fails ──────
# Edit this to match your real space if you want a better fallback.
ZONE_AREA_M2 = float(os.getenv("ZONE_AREA_M2", "50"))

# ── Fruin Level of Service thresholds (people / m²) ────────
# International crowd safety standard.
# These drive both zone_estimator and risk_engine.
FRUIN_FREE        = 0.5   # < 0.5  → free movement
FRUIN_RESTRICTED  = 1.0   # < 1.0  → restricted but comfortable
FRUIN_CONSTRAINED = 2.0   # < 2.0  → body contact possible
FRUIN_DANGEROUS   = 4.0   # < 4.0  → pushing / crowd pressure
                           # >= 4.0 → crush risk (Critical)

# Fruin capacities used by zone_estimator to derive headcounts:
FRUIN_SAFE_DENSITY = 1.0   # people/m² → safe operational capacity
FRUIN_MAX_DENSITY  = 1.5   # people/m² → absolute max capacity

# ── Time-to-breach prediction ──────────────────────────────
BREACH_RATE_WINDOW = 10   # recent readings used for fill-rate calc

# ── Camera ─────────────────────────────────────────────────
# CAMERA_SOURCE can be: a webcam index (0/1), an RTSP URL (CCTV),
# or an HTTP/MJPEG URL (phone camera app). It can also be changed at
# runtime from the website (saved in CAMERA_SETTINGS_PATH).
CAMERA_INDEX              = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_SOURCE             = os.getenv("CAMERA_SOURCE", str(CAMERA_INDEX))
CAMERA_MAX_WIDTH          = int(os.getenv("CAMERA_MAX_WIDTH", "960"))  # downscale big CCTV streams
CAMERA_SETTINGS_PATH      = os.getenv("CAMERA_SETTINGS_PATH", "camera_settings.json")
CAMERA_ADMIN_TOKEN        = os.getenv("CAMERA_ADMIN_TOKEN", "")  # optional: protects /camera/* changes

# ── Auto-recalibration ─────────────────────────────────────
AUTO_CALIBRATE_ENABLED         = os.getenv("AUTO_CALIBRATE_ENABLED", "true").lower() == "true"
AUTO_CALIBRATE_INTERVAL_SEC    = int(os.getenv("AUTO_CALIBRATE_INTERVAL_SEC", "30"))
AUTO_CALIBRATE_SKIP_WHEN_EMPTY = os.getenv("AUTO_CALIBRATE_SKIP_WHEN_EMPTY", "true").lower() == "true"
CALIBRATION_SMOOTHING          = float(os.getenv("CALIBRATION_SMOOTHING", "0.6"))  # 0 = no smoothing

YOLO_MODEL_PATH           = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
YOLO_SEG_MODEL_PATH       = os.getenv("YOLO_SEG_MODEL_PATH", "yolov8n-seg.pt")
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE", "0.4"))

# ── Floor detection ────────────────────────────────────────
# SegFormer-B5 trained on ADE20K (has a real "floor" class). Most accurate
# SegFormer size. About 340 MB, downloaded once on first start, then cached.
# Needs: pip install transformers
#   Faster, less accurate : nvidia/segformer-b2-finetuned-ade-512-512 (~110 MB)
#   Original method       : none  (whole frame minus top 20% minus objects)
FLOOR_MODEL_ID = os.getenv("FLOOR_MODEL_ID", "nvidia/segformer-b5-finetuned-ade-640-640")

# The floor itself rarely moves, so the heavy model only re-runs this often.
# Manual recalibration and camera changes always re-run it immediately.
FLOOR_REFRESH_SEC = int(os.getenv("FLOOR_REFRESH_SEC", "300"))

# Known real floor area in m2. When set, capacity uses this number instead of
# the camera-based estimate (the floor mask is still used for the preview).
# Can also be changed from the dashboard.
_area_env = os.getenv("ZONE_AREA_OVERRIDE_M2", "").strip()
ZONE_AREA_OVERRIDE_M2 = float(_area_env) if _area_env else None
ZONE_SETTINGS_PATH = os.getenv("ZONE_SETTINGS_PATH", "zone_settings.json")

# Used only when no real area is given: how many metres tall the camera's view
# of the floor is. Assumes a roughly top-down view; for angled cameras, enter
# the real area instead.
ZONE_VIEW_HEIGHT_M = float(os.getenv("ZONE_VIEW_HEIGHT_M", "5.0"))

# ── Risk engine — EMA smoothing ────────────────────────────
SMOOTHING_ALPHA = 0.7     # weight for current count vs history

# ── Risk engine — growth rate ──────────────────────────────
GROWTH_RATE_WINDOW = 3    # compare current count vs N readings ago

# ── Risk engine — persistence / hysteresis ─────────────────
HIGH_DENSITY_THRESHOLD   = 0.8   # occupancy ratio to enter high state
HIGH_DENSITY_EXIT        = 0.7   # occupancy ratio to exit high state
CRITICAL_PERSISTENCE_SEC = 10    # seconds in high state → Critical

# ── Risk engine — surge detection ──────────────────────────
SURGE_GROWTH_THRESHOLD = 3       # persons gained within growth window

# ── Risk score weights (must sum to 1.0) ───────────────────
WEIGHT_DENSITY     = 0.5
WEIGHT_GROWTH      = 0.3
WEIGHT_PERSISTENCE = 0.2

# ── Risk classification thresholds ─────────────────────────
RISK_THRESHOLD_LOW    = 0.4
RISK_THRESHOLD_MEDIUM = 0.6
RISK_THRESHOLD_HIGH   = 0.8

# ── Server ─────────────────────────────────────────────────
FLASK_PORT    = int(os.getenv("FLASK_PORT", "5001"))
DATABASE_PATH = os.getenv("DATABASE_PATH", "crowd_data.db")