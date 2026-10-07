"""
camera_source.py
Threaded camera reader for:
  - local webcam                 0 / 1
  - CCTV / NVR over RTSP         rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101
  - phone camera apps (MJPEG)    http://192.168.1.50:8080/video
Features: always-latest-frame, auto-reconnect, runtime source switching,
downscaling of large streams, and a probe() helper to test a URL.
"""
import os
import re
import threading
import time

# TCP is far more reliable than UDP for RTSP; low-latency flags avoid frame build-up.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)

import cv2  # noqa: E402

OPEN_TIMEOUT_MS = 6000
READ_TIMEOUT_MS = 6000


def parse_source(raw):
    """'0' -> 0 (webcam index); anything else stays a URL string."""
    if isinstance(raw, int):
        return raw
    raw = str(raw).strip()
    return int(raw) if raw.isdigit() else raw


def mask_source(src):
    """Hide the password so URLs are safe to show in the UI / logs."""
    if isinstance(src, int):
        return f"webcam:{src}"
    return re.sub(r"(://[^:/@\s]+):[^@\s]*@", r"\1:***@", str(src))


def open_capture(source):
    """Open a capture with timeouts. Returns cv2.VideoCapture or None."""
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
    else:
        try:
            cap = cv2.VideoCapture(
                source, cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                 cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS],
            )
        except (TypeError, AttributeError, cv2.error):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def probe(source, timeout=10):
    """
    Try to open `source` and read one frame, without touching the live camera.
    Returns {"ok": bool, "width":..., "height":..., "error": str|None}.
    """
    out = {"ok": False, "error": "Timed out connecting to the camera"}

    def work():
        cap = open_capture(parse_source(source))
        if cap is None:
            out.update(ok=False, error="Could not open stream "
                       "(wrong URL/path, wrong credentials, or camera unreachable)")
            return
        try:
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None:
                    out.update(ok=True, error=None,
                               width=int(frame.shape[1]), height=int(frame.shape[0]))
                    return
            out.update(ok=False, error="Connected, but no video frames were received")
        finally:
            cap.release()

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    return out


class CameraSource:
    def __init__(self, source, max_width=960, stale_after_sec=8, max_backoff_sec=15):
        self.source = parse_source(source)
        self.max_width = max_width
        self.stale_after_sec = stale_after_sec
        self.max_backoff_sec = max_backoff_sec

        self._frame = None
        self._frame_time = 0.0
        self._version = 0           # bumped whenever the source changes
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.status = "starting"    # starting | connecting | connected | reconnecting | stopped
        self.last_error = None

    # ---------- public ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self.status = "stopped"

    def set_source(self, source):
        """Switch camera at runtime (e.g. from the website). Reconnects immediately."""
        with self._lock:
            self.source = parse_source(source)
            self._frame = None
            self._frame_time = 0.0
            self._version += 1
            self.last_error = None
            self.status = "connecting"

    def get_frame(self):
        """Latest frame (copy), or None if not ready / stale."""
        with self._lock:
            if self._frame is None:
                return None
            if time.time() - self._frame_time > self.stale_after_sec:
                return None
            return self._frame.copy()

    @property
    def is_ready(self):
        return self.get_frame() is not None

    def info(self):
        age = round(time.time() - self._frame_time, 1) if self._frame_time else None
        return {
            "status": self.status,
            "ready": self.is_ready,
            "source": mask_source(self.source),
            "last_frame_age_sec": age,
            "last_error": self.last_error,
        }

    # ---------- internals ----------
    def _resize(self, frame):
        if self.max_width and frame.shape[1] > self.max_width:
            scale = self.max_width / frame.shape[1]
            frame = cv2.resize(frame, (self.max_width, int(frame.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
        return frame

    def _run(self):
        backoff = 1
        while not self._stop.is_set():
            version, source = self._version, self.source
            self.status = "connecting"
            cap = open_capture(source)

            if cap is None:
                self.last_error = "Cannot open camera source"
                print(f"[CAMERA] Cannot open {mask_source(source)} — retry in {backoff}s")
                self.status = "reconnecting"
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.max_backoff_sec)
                continue

            print(f"[CAMERA] Connected: {mask_source(source)}")
            self.status, self.last_error, backoff = "connected", None, 1

            while not self._stop.is_set() and version == self._version:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.last_error = "Stream interrupted"
                    print("[CAMERA] Read failed — reconnecting...")
                    break
                frame = self._resize(frame)
                with self._lock:
                    if version == self._version:
                        self._frame = frame
                        self._frame_time = time.time()

            cap.release()
            if version == self._version and not self._stop.is_set():
                self.status = "reconnecting"
                self._stop.wait(1)
