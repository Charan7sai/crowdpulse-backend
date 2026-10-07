# ──────────────────────────────────────────────
# Zone Estimator: floor detection and capacity
# ──────────────────────────────────────────────
# Floor detection
#   Optional: SegFormer trained on ADE20K, which has a real "floor" class.
#             Walls, desks, shelves, beds and objects on the floor are
#             excluded because the model labels them as something else.
#   Original : the first method (whole frame minus top 20% minus objects YOLO
#             recognises). This is the DEFAULT (FLOOR_MODEL_ID=none) and is
#             also the fallback if the SegFormer model cannot load.
#
# Capacity
#   area (m2) x Fruin density -> safe / max capacity.
#   Area comes from the user's real figure if one is set (most accurate),
#   otherwise a camera-view estimate (rough, assumes a near top-down view).
#
# People standing on the floor are counted as floor when ignore_people=True,
# so a crowd does not shrink the usable area.

import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

import config

FRUIN_SAFE_DENSITY = 1.0   # people/m2, comfortable operational limit
FRUIN_MAX_DENSITY  = 1.5   # people/m2, absolute physical maximum

# Fallback method only: COCO classes treated as "not floor"
NON_FLOOR_CLASSES = {0, 13, 56, 57, 58, 59, 60, 62, 63, 64, 66, 67, 73, 74, 75}

# ADE20K label names (first word) that count as walkable floor
FLOOR_LABELS = {"floor", "rug", "carpet"}


class ZoneEstimator:
    def __init__(self):
        print("[ZONE] Loading YOLOv8-seg model (person footprints)...")
        self._seg_model        = YOLO(config.YOLO_SEG_MODEL_PATH)
        self._usable_area_m2   = None
        self._safe_capacity    = None
        self._max_capacity     = None
        self._floor_mask       = None
        self._calibrated       = False
        self._pixels_per_metre = None
        self._lock             = threading.RLock()
        self._last_calibrated_at = None
        self._calibration_count  = 0
        self._method           = "yolov8-seg"
        self._area_source      = "estimated"
        self._coverage_pct     = None

        self._floor_net = None
        self._floor_proc = None
        self._floor_ids = []
        self._torch = None
        self._cached_floor = None
        self._cached_floor_at = 0.0
        self._load_floor_model()

    # ── floor model ───────────────────────────────────────

    def _load_floor_model(self):
        model_id = (config.FLOOR_MODEL_ID or "").strip()
        if not model_id or model_id.lower() == "none":
            print("[ZONE] Floor model disabled. Using object-subtraction fallback.")
            return
        try:
            import torch
            from transformers import (SegformerForSemanticSegmentation,
                                      SegformerImageProcessor)
            print(f"[ZONE] Loading floor model {model_id} (first run downloads it)...")
            self._floor_proc = SegformerImageProcessor.from_pretrained(model_id)
            self._floor_net = SegformerForSemanticSegmentation.from_pretrained(model_id).eval()
            self._torch = torch
            self._floor_ids = [
                int(i) for i, name in self._floor_net.config.id2label.items()
                if str(name).split(",")[0].strip().lower() in FLOOR_LABELS
            ]
            if not self._floor_ids:
                raise RuntimeError("model has no floor class")
            self._method = "segformer-ade20k"
            print(f"[ZONE] Floor model ready (class ids {self._floor_ids}).")
        except Exception as e:  # noqa: BLE001
            self._floor_net = None
            print(f"[ZONE] WARNING: floor model unavailable ({e}).")
            print("[ZONE] Using the old object-subtraction method instead.")
            print("[ZONE] Fix: pip install transformers   (internet needed once)")

    # ── public API ────────────────────────────────────────

    def calibrate(self, frame: np.ndarray,
                  pixels_per_metre: float = None,
                  ignore_people: bool = True,
                  smoothing: float = 0.0,
                  refresh_floor: bool = False) -> dict:
        """
        Thread-safe. Detects the floor, converts to area, sets capacity.

        ignore_people: people count as floor (a crowd does not shrink the area).
        smoothing:     0.0 use the new estimate, 0.6 blend 60% previous + 40% new.
                       Ignored when a manual area is set.
        refresh_floor: force the heavy floor model to run now instead of
                       reusing the last result (see FLOOR_REFRESH_SEC).
        """
        with self._lock:
            return self._calibrate_locked(frame, pixels_per_metre,
                                          ignore_people, smoothing, refresh_floor)

    def set_area_override(self, area_m2):
        """Set (or clear with None) the real floor area in m2."""
        with self._lock:
            config.ZONE_AREA_OVERRIDE_M2 = float(area_m2) if area_m2 else None
            if area_m2 and self._calibrated:
                self._apply_area(float(area_m2), "manual")
            return self.get_capacity()

    def get_capacity(self) -> dict | None:
        if not self._calibrated:
            return None
        return self._summary()

    def get_annotated_frame(self, frame: np.ndarray) -> np.ndarray:
        """Frame with the detected floor tinted green and outlined."""
        if self._floor_mask is None:
            return frame.copy()
        mask = self._floor_mask
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        tint = np.zeros_like(frame)
        tint[:, :] = (0, 200, 80)
        blended = cv2.addWeighted(frame, 0.6, tint, 0.4, 0)
        out = np.where(cv2.merge([mask] * 3) > 0, blended, frame)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 2)

        src = "entered by you" if self._area_source == "manual" else "estimated"
        lines = [
            f"Zone: {config.ZONE_NAME}",
            f"Floor area: {self._usable_area_m2} m2 ({src})",
            f"Safe capacity: {self._safe_capacity} people",
            f"Max capacity:  {self._max_capacity} people",
            f"Floor detection: {self._method}",
        ]
        for i, line in enumerate(lines):
            y = 30 + i * 28
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def is_calibrated(self) -> bool:
        return self._calibrated

    # ── calibration internals ─────────────────────────────

    def _calibrate_locked(self, frame, pixels_per_metre, ignore_people, smoothing,
                          refresh_floor=False):
        self._pixels_per_metre = pixels_per_metre
        h, w = frame.shape[:2]

        yolo_results = self._seg_model(frame, conf=0.3, verbose=False)

        if self._floor_net is not None:
            stale = (time.time() - self._cached_floor_at) > config.FLOOR_REFRESH_SEC
            if (refresh_floor or stale or self._cached_floor is None
                    or self._cached_floor.shape != (h, w)):
                print("[ZONE] Running floor model...")
                self._cached_floor = self._segformer_floor(frame)
                self._cached_floor_at = time.time()
            floor_mask = self._cached_floor.copy()
            if ignore_people:
                floor_mask = cv2.bitwise_or(
                    floor_mask, self._person_footprints(yolo_results, h, w))
            self._method = "segformer-ade20k"
        else:
            floor_mask = self._yolo_floor(yolo_results, h, w, ignore_people)
            self._method = "yolov8-seg"

        floor_mask = self._clean(floor_mask, legacy=(self._floor_net is None))
        self._floor_mask = floor_mask

        floor_px = int(np.count_nonzero(floor_mask))
        self._coverage_pct = round(100.0 * floor_px / (h * w), 1)

        override = config.ZONE_AREA_OVERRIDE_M2
        if override:
            area, source = float(override), "manual"
        else:
            area = self._pixels_to_m2(floor_px, h, w)
            source = "estimated"
            if (smoothing > 0 and self._calibrated
                    and self._usable_area_m2 is not None
                    and self._area_source == "estimated"):
                area = round(smoothing * self._usable_area_m2
                             + (1 - smoothing) * area, 2)

        self._apply_area(area, source)
        self._calibrated = True
        self._last_calibrated_at = time.time()
        self._calibration_count += 1

        print(f"[ZONE] Calibration complete ({self._method}, {source} area):")
        print(f"       Floor covers : {self._coverage_pct}% of the frame")
        print(f"       Usable area  : {self._usable_area_m2} m2")
        print(f"       Safe / max   : {self._safe_capacity} / {self._max_capacity} people")
        return self._summary()

    def _apply_area(self, area, source):
        self._usable_area_m2 = round(area, 2)
        self._area_source = source
        self._safe_capacity = max(1, int(self._usable_area_m2 * FRUIN_SAFE_DENSITY))
        self._max_capacity = max(1, int(self._usable_area_m2 * FRUIN_MAX_DENSITY))
        config.ZONE_AREA_M2       = self._usable_area_m2
        config.ZONE_SAFE_CAPACITY = self._safe_capacity
        config.ZONE_MAX_CAPACITY  = self._max_capacity

    def _segformer_floor(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self._floor_proc(images=rgb, return_tensors="pt")
        torch = self._torch
        with torch.no_grad():
            logits = self._floor_net(**inputs).logits
            up = torch.nn.functional.interpolate(
                logits, size=(h, w), mode="bilinear", align_corners=False)
            pred = up.argmax(dim=1)[0].cpu().numpy()
        return (np.isin(pred, self._floor_ids).astype(np.uint8)) * 255

    @staticmethod
    def _person_footprints(results, h, w):
        """Lower quarter of every detected person: where they stand on the floor."""
        out = np.zeros((h, w), dtype=np.uint8)
        for r in results:
            if r.masks is None:
                continue
            for i, mask_data in enumerate(r.masks.data):
                if int(r.boxes.cls[i]) != 0:
                    continue
                m = cv2.resize(mask_data.cpu().numpy(), (w, h)) > 0.5
                ys = np.where(m.any(axis=1))[0]
                if ys.size == 0:
                    continue
                cut = ys.min() + int(0.75 * (ys.max() - ys.min()))
                m[:cut, :] = False
                out[m] = 255
        return out

    @staticmethod
    def _yolo_floor(results, h, w, ignore_people):
        """Old method, kept as a fallback."""
        obstacle = np.zeros((h, w), dtype=np.uint8)
        for r in results:
            if r.masks is None:
                continue
            for i, mask_data in enumerate(r.masks.data):
                cls_id = int(r.boxes.cls[i])
                if ignore_people and cls_id == 0:
                    continue
                if cls_id in NON_FLOOR_CLASSES:
                    m = cv2.resize(mask_data.cpu().numpy(), (w, h))
                    obstacle = cv2.bitwise_or(obstacle, (m > 0.5).astype(np.uint8) * 255)
        floor = np.ones((h, w), dtype=np.uint8) * 255
        floor[:int(h * 0.20), :] = 0
        return cv2.bitwise_and(floor, cv2.bitwise_not(obstacle))

    @staticmethod
    def _clean(mask, legacy=False):
        """Fill small holes, drop speckles and tiny islands."""
        if legacy:
            # Exactly the original cleanup, so results match the first version.
            close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            opn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, opn)
        h, w = mask.shape[:2]
        k = max(5, int(min(h, w) * 0.02))
        close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k // 2),) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opn)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        min_area = 0.005 * h * w
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 255
        return keep

    def _pixels_to_m2(self, floor_pixels, frame_h, frame_w):
        if self._pixels_per_metre is not None:
            return round(floor_pixels / (self._pixels_per_metre ** 2), 2)
        metres_per_pixel = config.ZONE_VIEW_HEIGHT_M / frame_h
        return round(floor_pixels * (metres_per_pixel ** 2), 2)

    def _summary(self) -> dict:
        return {
            "zone_name"          : config.ZONE_NAME,
            "usable_area_m2"     : self._usable_area_m2,
            "safe_capacity"      : self._safe_capacity,
            "max_capacity"       : self._max_capacity,
            "fruin_safe_density" : FRUIN_SAFE_DENSITY,
            "fruin_max_density"  : FRUIN_MAX_DENSITY,
            "calibrated"         : self._calibrated,
            "pixels_per_metre"   : self._pixels_per_metre,
            "method"             : self._method,
            "area_source"        : self._area_source,
            "floor_coverage_pct" : self._coverage_pct,
            "last_calibrated_at" : self._last_calibrated_at,
            "calibration_count"  : self._calibration_count,
        }