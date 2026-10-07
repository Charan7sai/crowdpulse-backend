# ──────────────────────────────────────────────
# AI Detection Service — Flask + YOLOv8
# ──────────────────────────────────────────────
#
# Original endpoints (unchanged):
#   GET  /detect        — run YOLO on latest frame, return full risk assessment
#   GET  /history       — return recent crowd log entries
#   GET  /video_feed    — MJPEG stream, green boxes around persons only
#   GET  /zone          — AI-inferred zone capacity details
#   GET  /zone_preview  — MJPEG stream with detected floor highlighted in green
#   POST /recalibrate   — trigger fresh zone calibration on current frame
#
# New endpoints:
#   GET  /camera/status       — camera health + current (masked) source
#   GET  /camera/presets      — CCTV brands / phone apps the UI can offer
#   POST /camera/test         — test a camera URL / brand settings (no switching)
#   POST /camera/source       — switch camera (URL, or brand+ip+login)
#   POST /camera/discover     — try every known RTSP path for one camera
#   GET  /calibration_status  — auto-calibration state
#   POST /zone/area           — set the real floor area in m2 (or null to estimate)

from dotenv import load_dotenv
load_dotenv("my.env")   # must be before any config import

import hmac
import json
import os
import re
import threading
import time
from datetime import datetime, timezone

import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

import camera_presets
import config
import database
from auto_calibration import AutoCalibrator
from camera_source import CameraSource, mask_source, probe
from risk_engine import RiskEngine
from zone_estimator import ZoneEstimator

# ── Flask app ─────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Models ────────────────────────────────────
print("[APP] Loading YOLOv8 detection model...")
model = YOLO(config.YOLO_MODEL_PATH)
_model_lock = threading.Lock()   # one inference at a time (YOLO isn't thread-safe)
print("[APP] Detection model loaded.")

zone_estimator = ZoneEstimator()
risk_engine = RiskEngine()


def _load_zone_settings():
    """A real floor area entered on the dashboard survives restarts."""
    try:
        with open(config.ZONE_SETTINGS_PATH) as f:
            value = json.load(f).get("area_override_m2")
        if value:
            config.ZONE_AREA_OVERRIDE_M2 = float(value)
    except Exception:
        pass


def _save_zone_settings(value):
    try:
        with open(config.ZONE_SETTINGS_PATH, "w") as f:
            json.dump({"area_override_m2": value}, f)
    except Exception as e:
        print(f"[ZONE] Could not save settings: {e}")


_load_zone_settings()


# ── Camera (webcam / CCTV RTSP / phone) ───────
def _load_saved_source():
    """Source chosen from the website survives restarts."""
    try:
        with open(config.CAMERA_SETTINGS_PATH) as f:
            return json.load(f).get("source")
    except Exception:
        return None


def _save_source(source):
    try:
        with open(config.CAMERA_SETTINGS_PATH, "w") as f:
            json.dump({"source": source}, f)
        os.chmod(config.CAMERA_SETTINGS_PATH, 0o600)  # may contain camera password
    except Exception as e:
        print(f"[CAMERA] Could not save settings: {e}")


camera = CameraSource(
    _load_saved_source() or config.CAMERA_SOURCE,
    max_width=config.CAMERA_MAX_WIDTH,
)


# ── Detection helpers ─────────────────────────
def run_detection(frame):
    with _model_lock:
        return model(frame, conf=config.YOLO_CONFIDENCE_THRESHOLD, verbose=False)


def count_persons(frame):
    n = 0
    for r in run_detection(frame):
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                n += 1
    return n


def current_person_count():
    frame = camera.get_frame()
    return None if frame is None else count_persons(frame)


# ── Calibration (shared by button + auto) ─────
def do_calibration(reason):
    frame = camera.get_frame()
    if frame is None:
        raise RuntimeError("Camera not ready")

    print(f"[ZONE] Calibration ({reason})...")
    # People are treated as floor, so calibrating during a crowd is safe.
    # Manual / camera-change = fresh estimate. Auto = smoothed to avoid jitter.
    fresh = reason in ("manual", "camera_changed")
    result = zone_estimator.calibrate(
        frame,
        ignore_people=True,
        smoothing=0.0 if fresh else config.CALIBRATION_SMOOTHING,
        refresh_floor=fresh,
    )

    if config.ZONE_MAX_CAPACITY is None or config.ZONE_SAFE_CAPACITY is None:
        area = config.ZONE_AREA_M2
        config.ZONE_SAFE_CAPACITY = max(1, int(area * 1.0))
        config.ZONE_MAX_CAPACITY = max(1, int(area * 1.5))
        print(f"[ZONE] AI calibration incomplete — fallback area {area} m²")
    return result


calibrator = AutoCalibrator(
    calibrate_fn=do_calibration,
    people_count_fn=current_person_count,
    is_calibrated_fn=zone_estimator.is_calibrated,
    interval_sec=config.AUTO_CALIBRATE_INTERVAL_SEC,
    enabled=config.AUTO_CALIBRATE_ENABLED,
    skip_when_empty=config.AUTO_CALIBRATE_SKIP_WHEN_EMPTY,
)


# ── Helpers for /camera/* routes ──────────────
def _admin_ok():
    token = config.CAMERA_ADMIN_TOKEN
    if not token:
        return True
    return hmac.compare_digest(request.headers.get("X-Admin-Token", ""), token)


def _resolve_source(body):
    """Turn the request body into a camera source (URL string or webcam index)."""
    url = str(body.get("url") or "").strip()
    if url:
        if url.isdigit():
            return int(url)
        if not re.match(r"^(rtsp|rtsps|http|https)://", url, re.I):
            raise ValueError("URL must start with rtsp://, rtsps://, http:// or https://")
        return url
    if body.get("brand"):
        return camera_presets.build_url(
            brand=body["brand"],
            ip=body.get("ip", ""),
            user=body.get("user", ""),
            password=body.get("password", ""),
            port=body.get("port"),
            channel=body.get("channel", 1),
            stream=body.get("stream", "main"),
            path=body.get("path"),
        )
    raise ValueError("Provide either 'url' or 'brand' + 'ip'")


# ── Endpoints ─────────────────────────────────

@app.route("/detect")
def detect():
    frame = camera.get_frame()
    if frame is None:
        return jsonify({"error": "Camera not ready. Check the camera source / connection."}), 503

    raw_count = count_persons(frame)

    assessment = risk_engine.evaluate(raw_count)
    data = assessment.to_dict()
    data["timestamp"] = datetime.fromtimestamp(
        data["timestamp"], tz=timezone.utc
    ).isoformat()

    threading.Thread(target=database.log_reading, args=(data,), daemon=True).start()
    return jsonify(data)


@app.route("/history")
def history():
    minutes = request.args.get("minutes", 2, type=int)
    minutes = max(1, min(minutes, 60))
    return jsonify(database.get_history(minutes))


def _mjpeg(parts):
    return Response(parts, mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            annotated = frame.copy()
            for r in run_detection(frame):
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated, f"Person {conf:.0%}", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            ret, jpeg = cv2.imencode(".jpg", annotated)
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpeg.tobytes() + b"\r\n")
            time.sleep(0.05)

    return _mjpeg(generate())


@app.route("/zone")
def zone():
    info = zone_estimator.get_capacity()
    if info is None:
        return jsonify({"error": "Zone not yet calibrated. Wait a moment and retry."}), 503
    return jsonify(info)


@app.route("/zone_preview")
def zone_preview():
    def generate():
        while True:
            frame = camera.get_frame()
            if frame is None or not zone_estimator.is_calibrated():
                time.sleep(0.1)
                continue
            annotated = zone_estimator.get_annotated_frame(frame)
            ret, jpeg = cv2.imencode(".jpg", annotated)
            if not ret:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpeg.tobytes() + b"\r\n")
            time.sleep(0.1)

    return _mjpeg(generate())


@app.route("/zone/area", methods=["POST"])
def zone_area():
    """
    Enter the real floor area in m2 for accurate capacity.
    Send {"area_m2": null} to go back to the camera-based estimate.
    """
    if not _admin_ok():
        return jsonify({"error": "Unauthorized"}), 401
    raw = (request.get_json(silent=True) or {}).get("area_m2")
    value = None
    if raw not in (None, "", 0):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return jsonify({"error": "area_m2 must be a number"}), 400
        if not (1 <= value <= 100000):
            return jsonify({"error": "area_m2 must be between 1 and 100000"}), 400

    zone_estimator.set_area_override(value)
    _save_zone_settings(value)
    if value is None:
        calibrator.request_now("camera_changed")   # go back to the estimate now
    return jsonify({"status": "ok", "area_m2": value, "zone": zone_estimator.get_capacity()})


@app.route("/recalibrate", methods=["POST"])
def recalibrate():
    """The dashboard Recalibrate button. Same URL and response shape as before."""
    print("[ZONE] Manual recalibration triggered from dashboard...")
    ok, result = calibrator.run("manual")
    if not ok:
        code = 503 if "Camera not ready" in str(result) else 409
        return jsonify({"error": result}), code
    return jsonify({"status": "recalibrated", "zone": result})


@app.route("/calibration_status")
def calibration_status():
    return jsonify(calibrator.status())


# ── Camera management (for the website) ───────

@app.route("/camera/status")
def camera_status():
    return jsonify({**camera.info(), "calibration": calibrator.status()})


@app.route("/camera/presets")
def camera_presets_list():
    return jsonify(camera_presets.list_presets())


@app.route("/camera/test", methods=["POST"])
def camera_test():
    if not _admin_ok():
        return jsonify({"error": "Unauthorized"}), 401
    try:
        source = _resolve_source(request.get_json(silent=True) or {})
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    result = probe(source, timeout=12)
    result["source"] = mask_source(source)
    return jsonify(result), (200 if result["ok"] else 422)


@app.route("/camera/source", methods=["GET", "POST"])
def camera_source_route():
    if request.method == "GET":
        return jsonify(camera.info())

    if not _admin_ok():
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    try:
        source = _resolve_source(body)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not body.get("skip_test"):
        result = probe(source, timeout=12)
        if not result["ok"]:
            return jsonify({"error": result["error"], "source": mask_source(source)}), 422

    camera.set_source(source)
    _save_source(source if isinstance(source, int) else str(source))
    calibrator.request_now("camera_changed")   # new scene → recalibrate right away
    print(f"[CAMERA] Switched to {mask_source(source)}")
    return jsonify({"status": "switched", **camera.info()})


@app.route("/camera/discover", methods=["POST"])
def camera_discover():
    if not _admin_ok():
        return jsonify({"error": "Unauthorized"}), 401
    body = request.get_json(silent=True) or {}
    if not body.get("ip"):
        return jsonify({"error": "'ip' is required"}), 400
    try:
        result = camera_presets.discover(
            ip=body["ip"],
            user=body.get("user", ""),
            password=body.get("password", ""),
            port=body.get("port"),
            channel=body.get("channel", 1),
            brand=body.get("brand"),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(result)


# ── Startup ───────────────────────────────────

if __name__ == "__main__":
    print("[APP] Initialising database...")
    database.init_db()

    print(f"[APP] Starting camera: {mask_source(camera.source)}")
    camera.start()
    calibrator.start()

    print(f"[APP] Server starting on port {config.FLASK_PORT}")
    app.run(port=config.FLASK_PORT, threaded=True)