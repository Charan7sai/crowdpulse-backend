"""
auto_calibration.py
Keeps the zone capacity up to date while the crowd changes.

Policy (every `interval_sec`):
  - Never calibrated yet            -> calibrate (as soon as the camera is ready)
  - People are in the frame         -> recalibrate (scene is changing)
  - Floor is empty & already done   -> SKIP (nothing to correct)
  - Camera was just switched        -> calibrate immediately, regardless

The manual dashboard button uses the same run() method, so manual and auto
runs can never overlap.
"""
import threading
import time
from datetime import datetime, timezone


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


class AutoCalibrator:
    def __init__(self, calibrate_fn, people_count_fn, is_calibrated_fn,
                 interval_sec=30, enabled=True, skip_when_empty=True,
                 first_run_delay_sec=3, retry_sec=3):
        """
        calibrate_fn(reason)  -> performs calibration, returns a result dict
                                 (raise an exception on failure)
        people_count_fn()     -> int people currently visible, or None if unknown
        is_calibrated_fn()    -> bool
        """
        self.calibrate_fn = calibrate_fn
        self.people_count_fn = people_count_fn
        self.is_calibrated_fn = is_calibrated_fn
        self.interval_sec = interval_sec
        self.enabled = enabled
        self.skip_when_empty = skip_when_empty
        self.retry_sec = retry_sec

        self._run_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._next_due = time.time() + first_run_delay_sec
        self._force = False
        self._force_reason = "auto"

        self.last_run = None
        self.last_reason = None
        self.last_error = None
        self.last_skip = None
        self.last_people_count = None
        self.run_count = 0

    # ---------- public ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def request_now(self, reason="camera_changed"):
        """Ask for an immediate calibration that ignores the 'empty floor' skip."""
        self._force, self._force_reason, self._next_due = True, reason, 0

    def run(self, reason="manual"):
        """Used by the manual button and the scheduler. Returns (ok, result|message)."""
        if not self._run_lock.acquire(blocking=False):
            return False, "Calibration already in progress"
        try:
            result = self.calibrate_fn(reason)
            self.last_run, self.last_reason, self.last_error = _now_iso(), reason, None
            self.run_count += 1
            self._force = False
            self._next_due = time.time() + self.interval_sec
            return True, result
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)
            print(f"[CALIBRATION] {reason} run failed: {e}")
            return False, str(e)
        finally:
            self._run_lock.release()

    def status(self):
        return {
            "auto_enabled": self.enabled,
            "interval_sec": self.interval_sec,
            "skip_when_empty": self.skip_when_empty,
            "last_run": self.last_run,
            "last_reason": self.last_reason,
            "last_error": self.last_error,
            "last_skip": self.last_skip,
            "last_people_count": self.last_people_count,
            "run_count": self.run_count,
            "next_check_in_sec": max(0, int(self._next_due - time.time())) if self.enabled else None,
        }

    # ---------- internals ----------
    def _loop(self):
        while not self._stop.wait(1):
            if not self.enabled or time.time() < self._next_due:
                continue

            force = self._force
            if not force and self.is_calibrated_fn():
                try:
                    count = self.people_count_fn()
                except Exception as e:  # noqa: BLE001
                    print(f"[CALIBRATION] people count failed: {e}")
                    count = None
                self.last_people_count = count
                if count is None:                      # camera not ready, try soon
                    self._next_due = time.time() + self.retry_sec
                    continue
                if count == 0 and self.skip_when_empty:
                    self.last_skip = {"at": _now_iso(), "reason": "floor empty — no recalibration needed"}
                    self._next_due = time.time() + self.interval_sec
                    continue

            ok, msg = self.run(self._force_reason if force else "auto")
            if ok:
                print(f"[CALIBRATION] {self.last_reason} run ok")
            else:
                self._next_due = time.time() + self.retry_sec
