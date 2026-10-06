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
CAMERA_INDEX              = int(os.getenv("CAMERA_INDEX", "0"))
YOLO_MODEL_PATH           = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
YOLO_SEG_MODEL_PATH       = os.getenv("YOLO_SEG_MODEL_PATH", "yolov8n-seg.pt")
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE", "0.4"))

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