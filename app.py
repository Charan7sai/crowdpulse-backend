# ──────────────────────────────────────────────
# AI Detection Service — Flask + YOLOv8
# ──────────────────────────────────────────────
#
# Endpoints:
#   GET  /detect        — run YOLO on latest frame, return full risk assessment
#   GET  /history       — return recent crowd log entries
#   GET  /video_feed    — MJPEG stream, green boxes around persons only
#   GET  /zone          — AI-inferred zone capacity details
#   GET  /zone_preview  — MJPEG stream with detected floor highlighted in green
#   POST /recalibrate   — trigger fresh zone calibration on current frame

from dotenv import load_dotenv
load_dotenv("my.env")   # must be before any config import

import threading
import time
from datetime import datetime, timezone

import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

import config
import database
from risk_engine import RiskEngine
from zone_estimator import ZoneEstimator

# ── Flask app ─────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow Next.js frontend to call backend directly if needed

# ── Models ────────────────────────────────────
print("[APP] Loading YOLOv8 detection model...")
model = YOLO(config.YOLO_MODEL_PATH)
print("[APP] Detection model loaded.")

# ── Zone estimator (AI floor detection) ───────
zone_estimator = ZoneEstimator()

# ── Risk engine (stateful singleton) ──────────
risk_engine = RiskEngine()

# ── Shared camera frame ───────────────────────
_latest_frame = None
_frame_lock   = threading.Lock()
_camera_ready = threading.Event()


# ── Camera capture thread ─────────────────────

def _camera_loop():
    """
    Background daemon thread.
    Continuously grabs frames from the webcam.

    Calibration strategy:
    - Waits 30 frames (~3 seconds) before calibrating
    - Prints warning to move away from camera
    - Uses last collected frame (room should be empty by then)
    - Falls back to ZONE_AREA_M2 from my.env if AI fails
    """
    global _latest_frame

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {config.CAMERA_INDEX}.")
        print(f"[ERROR] Try changing CAMERA_INDEX in my.env (0 or 1).")
        return

    _camera_ready.set()
    print(f"[INFO] Camera {config.CAMERA_INDEX} opened — capture thread running")

    calibrated      = False
    frame_count     = 0
    CALIBRATE_AFTER = 30   # ~3 seconds at ~10fps
    warned          = False

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # ── Zone calibration logic ─────────────
        if not calibrated:
            frame_count += 1

            if frame_count == 1:
                print("[ZONE] Waiting 3 seconds before calibration...")
                print("[ZONE] *** PLEASE MOVE AWAY FROM THE CAMERA ***")

            if frame_count == CALIBRATE_AFTER // 2 and not warned:
                print("[ZONE] Calibrating in ~1.5 seconds — make sure area is empty...")
                warned = True

            if frame_count >= CALIBRATE_AFTER:
                print("[ZONE] Calibrating zone now...")
                zone_estimator.calibrate(frame)
                calibrated = True

                # Safety fallback if AI returned None
                if config.ZONE_MAX_CAPACITY is None or config.ZONE_SAFE_CAPACITY is None:
                    area = config.ZONE_AREA_M2
                    config.ZONE_SAFE_CAPACITY = max(1, int(area * 1.0))
                    config.ZONE_MAX_CAPACITY  = max(1, int(area * 1.5))
                    print(f"[ZONE] AI calibration incomplete — using area fallback")
                    print(f"[ZONE] Fallback area  : {area} m²")
                    print(f"[ZONE] Fallback safe  : {config.ZONE_SAFE_CAPACITY} people")
                    print(f"[ZONE] Fallback max   : {config.ZONE_MAX_CAPACITY} people")
        # ───────────────────────────────────────

        with _frame_lock:
            _latest_frame = frame


def _start_camera():
    t = threading.Thread(target=_camera_loop, daemon=True)
    t.start()
    _camera_ready.wait(timeout=5)


# ── Endpoints ─────────────────────────────────

@app.route("/detect")
def detect():
    """
    Run YOLOv8 detection on the latest frame.
    Returns full crowd risk assessment including:
    - occupancy vs AI-inferred capacity
    - people/m2 with Fruin LoS classification
    - time-to-breach predictions
    - risk score and level
    """
    with _frame_lock:
        frame = _latest_frame

    if frame is None:
        return jsonify({"error": "Camera not ready. Check CAMERA_INDEX in my.env"}), 503

    # Run YOLO person detection
    results = model(
        frame,
        conf=config.YOLO_CONFIDENCE_THRESHOLD,
        verbose=False
    )

    # Count persons only (COCO class 0)
    raw_count = 0
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                raw_count += 1

    # Evaluate risk using AI-inferred capacity
    assessment = risk_engine.evaluate(raw_count)
    data = assessment.to_dict()

    # Format timestamp for JSON
    data["timestamp"] = datetime.fromtimestamp(
        data["timestamp"], tz=timezone.utc
    ).isoformat()

    # Log to database asynchronously (non-blocking)
    threading.Thread(
        target=database.log_reading,
        args=(data,),
        daemon=True
    ).start()

    return jsonify(data)


@app.route("/history")
def history():
    """Return recent crowd log entries for trend charts."""
    minutes = request.args.get("minutes", 2, type=int)
    minutes = max(1, min(minutes, 60))
    rows = database.get_history(minutes)
    return jsonify(rows)


@app.route("/video_feed")
def video_feed():
    """
    MJPEG stream of live camera feed.
    Shows green bounding boxes around persons only.
    No boxes around objects, furniture, etc.
    """
    def generate():
        while True:
            with _frame_lock:
                frame = _latest_frame

            if frame is None:
                time.sleep(0.1)
                continue

            # Run YOLO detection
            results = model(
                frame,
                conf=config.YOLO_CONFIDENCE_THRESHOLD,
                verbose=False
            )

            # Draw boxes around persons only (class 0)
            annotated = frame.copy()
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:  # person only
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        # Green box
                        cv2.rectangle(
                            annotated,
                            (x1, y1), (x2, y2),
                            (0, 255, 0), 2
                        )
                        # Label
                        cv2.putText(
                            annotated,
                            f"Person {conf:.0%}",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 2
                        )

            ret, jpeg = cv2.imencode(".jpg", annotated)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
            time.sleep(0.05)  # ~20 fps cap

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/zone")
def zone():
    """
    Returns the AI-inferred zone capacity.
    Computed at startup by YOLOv8-seg floor detection.
    ZONE_MAX_CAPACITY and ZONE_SAFE_CAPACITY are never
    manually set — fully AI-driven from camera feed.
    """
    info = zone_estimator.get_capacity()
    if info is None:
        return jsonify({
            "error": "Zone not yet calibrated. Wait a moment and retry."
        }), 503
    return jsonify(info)


@app.route("/zone_preview")
def zone_preview():
    """
    MJPEG stream with AI-detected floor highlighted in green.
    Shows zone name, area, and capacity numbers on screen.
    Best demo visual — proves AI floor detection is working.
    """
    def generate():
        while True:
            with _frame_lock:
                frame = _latest_frame

            if frame is None or not zone_estimator.is_calibrated():
                time.sleep(0.1)
                continue

            annotated = zone_estimator.get_annotated_frame(frame)
            ret, jpeg = cv2.imencode(".jpg", annotated)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
            time.sleep(0.1)  # ~10 fps

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/recalibrate", methods=["POST"])
def recalibrate():
    """
    Trigger a fresh zone calibration on the current frame.
    Use this when the area is empty before an event.
    The dashboard Recalibrate button calls this endpoint.
    """
    with _frame_lock:
        frame = _latest_frame

    if frame is None:
        return jsonify({"error": "Camera not ready"}), 503

    print("[ZONE] Manual recalibration triggered from dashboard...")
    result = zone_estimator.calibrate(frame)

    # Safety fallback
    if config.ZONE_MAX_CAPACITY is None or config.ZONE_SAFE_CAPACITY is None:
        area = config.ZONE_AREA_M2
        config.ZONE_SAFE_CAPACITY = max(1, int(area * 1.0))
        config.ZONE_MAX_CAPACITY  = max(1, int(area * 1.5))
        print(f"[ZONE] Fallback used — safe: {config.ZONE_SAFE_CAPACITY}, "
              f"max: {config.ZONE_MAX_CAPACITY}")

    print(f"[ZONE] Recalibration complete: {result}")
    return jsonify({"status": "recalibrated", "zone": result})


# ── Startup ───────────────────────────────────

if __name__ == "__main__":
    print("[APP] Initialising database...")
    database.init_db()

    print("[APP] Starting camera...")
    _start_camera()

    print(f"[APP] Server starting on port {config.FLASK_PORT}")
    print(f"[APP] Endpoints:")
    print(f"      http://localhost:{config.FLASK_PORT}/detect")
    print(f"      http://localhost:{config.FLASK_PORT}/history")
    print(f"      http://localhost:{config.FLASK_PORT}/zone")
    print(f"      http://localhost:{config.FLASK_PORT}/video_feed      (browser)")
    print(f"      http://localhost:{config.FLASK_PORT}/zone_preview    (browser)")
    print(f"      POST http://localhost:{config.FLASK_PORT}/recalibrate")

    app.run(port=config.FLASK_PORT, threaded=True)