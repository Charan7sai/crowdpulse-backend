# ──────────────────────────────────────────────
# Zone Estimator — YOLOv8-seg AI floor detection
# ──────────────────────────────────────────────
# Uses YOLOv8 segmentation to identify walkable
# floor area from an empty reference frame, then
# derives safe & max capacity using Fruin LoS
# (Level of Service) international safety standard.
#
# Called once at startup by app.py with the first
# camera frame. Results are pushed into config so
# risk_engine.py picks them up automatically.

import cv2
import numpy as np
from ultralytics import YOLO

import config

# ── Fruin Level of Service ─────────────────────
FRUIN_SAFE_DENSITY = 1.0   # people/m² — comfortable operational limit
FRUIN_MAX_DENSITY  = 1.5   # people/m² — absolute physical maximum

# ── COCO class IDs that are NOT walkable floor ─
# YOLOv8 detects these and subtracts them from floor area
NON_FLOOR_CLASSES = {
    0,   # person
    13,  # bench
    56,  # chair
    57,  # couch
    58,  # potted plant
    59,  # bed
    60,  # dining table
    62,  # tv / monitor
    63,  # laptop
    64,  # mouse
    66,  # keyboard
    67,  # cell phone
    73,  # book
    74,  # clock
    75,  # vase
}


class ZoneEstimator:
    """
    Estimates usable floor area using YOLOv8-seg.

    Usage:
        estimator = ZoneEstimator()
        result = estimator.calibrate(empty_frame)
        # config.ZONE_SAFE_CAPACITY and ZONE_MAX_CAPACITY
        # are now updated automatically.
    """

    def __init__(self):
        print("[ZONE] Loading YOLOv8-seg model...")
        self._seg_model        = YOLO(config.YOLO_SEG_MODEL_PATH)
        self._usable_area_m2   = None
        self._safe_capacity    = None
        self._max_capacity     = None
        self._floor_mask       = None
        self._calibrated       = False
        self._pixels_per_metre = None
        print("[ZONE] Seg model loaded.")

    # ── public API ────────────────────────────────────────

    def calibrate(self, empty_frame: np.ndarray,
                  pixels_per_metre: float = None) -> dict:
        """
        Run once at startup on a reference camera frame.
        Ideally the room should be empty or near-empty.

        pixels_per_metre: optional real-world calibration.
            If None, uses heuristic (frame height ≈ 5 metres).
            This is good enough for CCTV mounted at 3–6m height.

        Returns a dict summary and updates config values.
        """
        self._pixels_per_metre = pixels_per_metre
        h, w = empty_frame.shape[:2]

        print("[ZONE] Running segmentation on reference frame...")

        # 1. Run YOLOv8-seg on the reference frame
        results = self._seg_model(
            empty_frame,
            conf=0.3,      # lower threshold — catch more obstacles
            verbose=False
        )

        # 2. Build obstacle mask from all non-floor detected objects
        obstacle_mask = np.zeros((h, w), dtype=np.uint8)

        for r in results:
            if r.masks is None:
                continue
            for i, mask_data in enumerate(r.masks.data):
                cls_id = int(r.boxes.cls[i])
                if cls_id in NON_FLOOR_CLASSES:
                    m = mask_data.cpu().numpy()
                    m = cv2.resize(m, (w, h))
                    obstacle_mask = cv2.bitwise_or(
                        obstacle_mask,
                        (m > 0.5).astype(np.uint8) * 255
                    )

        # 3. Floor mask = full frame minus obstacles minus ceiling zone
        floor_mask = np.ones((h, w), dtype=np.uint8) * 255

        # Remove top 20% — ceiling, upper walls, not walkable
        floor_mask[:int(h * 0.20), :] = 0

        # Remove detected non-floor objects
        floor_mask = cv2.bitwise_and(
            floor_mask,
            cv2.bitwise_not(obstacle_mask)
        )

        # 4. Morphological cleanup — fill small holes, remove noise
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (20, 20)
        )
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (10, 10)
        )
        floor_mask = cv2.morphologyEx(
            floor_mask, cv2.MORPH_CLOSE, kernel_close
        )
        floor_mask = cv2.morphologyEx(
            floor_mask, cv2.MORPH_OPEN, kernel_open
        )

        self._floor_mask = floor_mask

        # 5. Convert pixel count to real-world m²
        floor_pixels = int(np.count_nonzero(floor_mask))
        self._usable_area_m2 = self._pixels_to_m2(floor_pixels, h, w)

        # 6. Derive capacity using Fruin safety standards
        self._safe_capacity = max(1, int(
            self._usable_area_m2 * FRUIN_SAFE_DENSITY
        ))
        self._max_capacity = max(1, int(
            self._usable_area_m2 * FRUIN_MAX_DENSITY
        ))

        # 7. Push AI-inferred values into config
        #    Risk engine will use these from now on
        config.ZONE_AREA_M2       = self._usable_area_m2
        config.ZONE_SAFE_CAPACITY = self._safe_capacity
        config.ZONE_MAX_CAPACITY  = self._max_capacity

        self._calibrated = True

        summary = self._summary()
        print(f"[ZONE] Calibration complete:")
        print(f"       Usable area  : {self._usable_area_m2} m²")
        print(f"       Safe capacity: {self._safe_capacity} people "
              f"({FRUIN_SAFE_DENSITY} p/m²)")
        print(f"       Max capacity : {self._max_capacity} people "
              f"({FRUIN_MAX_DENSITY} p/m²)")

        return summary

    def get_capacity(self) -> dict | None:
        """Returns current estimated capacity dict, or None if not calibrated."""
        if not self._calibrated:
            return None
        return self._summary()

    def get_annotated_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns a copy of the frame with the detected floor area
        highlighted in green. Used by the /zone_preview endpoint.
        Great for demos — shows investors the AI floor detection.
        """
        if self._floor_mask is None:
            return frame.copy()

        overlay = frame.copy()
        green = np.zeros_like(frame)
        green[:, :] = (0, 200, 80)  # BGR green

        mask_3ch = cv2.merge([self._floor_mask] * 3)
        overlay = np.where(
            mask_3ch > 0,
            cv2.addWeighted(frame, 0.55, green, 0.45, 0),
            frame
        )

        # Draw capacity info on top
        lines = [
            f"Zone: {config.ZONE_NAME}",
            f"Detected area: {self._usable_area_m2} m2",
            f"Safe capacity: {self._safe_capacity} people",
            f"Max capacity:  {self._max_capacity} people",
            f"Method: YOLOv8-seg (Fruin LoS)",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                overlay, line,
                (10, 30 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA
            )

        return overlay

    def is_calibrated(self) -> bool:
        return self._calibrated

    # ── private helpers ───────────────────────────────────

    def _pixels_to_m2(self, floor_pixels: int,
                       frame_h: int, frame_w: int) -> float:
        """
        Converts pixel area to square metres.
        Uses real calibration if provided, else heuristic.
        """
        if self._pixels_per_metre is not None:
            # Calibrated path: user told us pixels per metre
            px_per_m2 = self._pixels_per_metre ** 2
            area = floor_pixels / px_per_m2
        else:
            # Heuristic path: assume camera height ≈ 5 real-world metres.
            # Works well for ceiling-mounted CCTV at 3–6m height.
            metres_per_pixel = 5.0 / frame_h
            area = floor_pixels * (metres_per_pixel ** 2)

        return round(area, 2)

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
            "method"             : "yolov8-seg",
        }