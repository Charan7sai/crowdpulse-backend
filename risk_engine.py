# ──────────────────────────────────────────────
# Stateful Risk Engine — Real-Time Crowd Risk Assessment
# ──────────────────────────────────────────────
#
# Evaluates crowd risk using three factors:
#   1. Occupancy ratio  (current / max capacity, area-based)
#   2. Growth rate      (change over a sliding window)
#   3. Persistence      (duration in high-density state)
#
# Also computes:
#   - People per m² with Fruin Level of Service classification
#   - Fill rate (people entering per minute)
#   - Time-to-breach predictions for safe & max capacity
#
# All capacity values come from config, which is updated
# at startup by zone_estimator.py using AI floor detection.

import time
from collections import deque

import config


class RiskAssessment:
    """Data container for a single risk evaluation."""

    __slots__ = (
        # Raw detection
        "current_count",
        "smoothed_count",
        # Capacity & occupancy
        "zone_name",
        "zone_max_capacity",
        "zone_safe_capacity",
        "zone_area_m2",
        "occupancy_ratio",           # current / max capacity (0.0–1.5+)
        "occupancy_pct",             # human-readable percentage string
        "safe_slots_left",           # safe_capacity - current_count
        # Area-based density
        "density_per_m2",            # people per square metre
        "fruin_level",               # Free | Restricted | Constrained | Dangerous | Critical
        # Dynamics
        "growth_rate",               # people added vs N readings ago
        "fill_rate_per_min",         # projected people entering per minute
        # Time-to-breach predictions
        "time_to_safe_breach_sec",   # seconds until safe_capacity is hit
        "time_to_max_breach_sec",    # seconds until max_capacity is hit
        "time_to_safe_breach_label", # human-readable e.g. "4 min 20 sec"
        "time_to_max_breach_label",
        # Risk scoring
        "risk_score",
        "risk_level",
        "surge_flag",
        "duration_in_high_state",
        "timestamp",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}


class RiskEngine:
    """
    Stateful risk engine.
    Call evaluate(raw_count) on every detection cycle.
    """

    def __init__(self):
        self._alpha = config.SMOOTHING_ALPHA

        # Circular buffer of (timestamp, raw_count) — for fill rate & growth
        self._history: deque = deque(
            maxlen=max(config.BREACH_RATE_WINDOW, 10)
        )

        # EMA state
        self._smoothed_count: float | None = None

        # Persistence tracking
        self._high_state_start: float | None = None
    # Guard: if AI hasn't calibrated yet, use area-based fallback
    zone_max  = config.ZONE_MAX_CAPACITY  or max(1, int(config.ZONE_AREA_M2 * 1.5))
    zone_safe = config.ZONE_SAFE_CAPACITY or max(1, int(config.ZONE_AREA_M2 * 1.0))
    # ── public API ────────────────────────────────────────

    def evaluate(self, raw_count: int) -> RiskAssessment:
        now = time.time()

        # 1. EMA smoothing
        if self._smoothed_count is None:
            self._smoothed_count = float(raw_count)
        else:
            self._smoothed_count = (
                self._alpha * raw_count
                + (1 - self._alpha) * self._smoothed_count
            )

        # 2. Record history
        self._history.append((now, raw_count))

        # 3. Read current config values
        #    (zone_estimator updates these at startup)
        zone_max  = config.ZONE_MAX_CAPACITY
        zone_safe = config.ZONE_SAFE_CAPACITY
        zone_area = config.ZONE_AREA_M2

        # 4. Occupancy & area metrics
        occupancy_ratio = self._smoothed_count / zone_max  # can exceed 1.0
        occupancy_pct   = f"{min(occupancy_ratio * 100, 999):.1f}%"
        safe_slots_left = zone_safe - raw_count
        density_per_m2  = round(self._smoothed_count / zone_area, 3)
        fruin_level     = self._fruin_classify(density_per_m2)

        # 5. Growth rate (vs N readings ago)
        window = config.GROWTH_RATE_WINDOW
        if len(self._history) > window:
            old_count = self._history[-1 - window][1]
            growth_rate = float(raw_count - old_count)
        else:
            growth_rate = 0.0

        surge_flag = growth_rate >= config.SURGE_GROWTH_THRESHOLD

        # 6. Fill rate (people / minute) via linear slope over recent window
        fill_rate_per_min = self._compute_fill_rate()

        # 7. Time-to-breach predictions
        t_safe, t_max = self._compute_breach_times(
            raw_count, fill_rate_per_min, zone_safe, zone_max
        )

        # 8. Persistence timer with hysteresis
        if occupancy_ratio >= config.HIGH_DENSITY_THRESHOLD:
            if self._high_state_start is None:
                self._high_state_start = now
            duration_in_high_state = round(now - self._high_state_start, 1)
        else:
            if occupancy_ratio < config.HIGH_DENSITY_EXIT:
                self._high_state_start = None
            duration_in_high_state = (
                round(now - self._high_state_start, 1)
                if self._high_state_start is not None
                else 0.0
            )

        # 9. Normalised sub-scores
        persistence_factor = min(
            duration_in_high_state / config.CRITICAL_PERSISTENCE_SEC, 1.0
        )
        normalized_growth = min(
            max(growth_rate, 0) / config.SURGE_GROWTH_THRESHOLD, 1.0
        )

        # 10. Compound risk score
        risk_score = (
            config.WEIGHT_DENSITY      * min(occupancy_ratio, 1.0)
            + config.WEIGHT_GROWTH     * normalized_growth
            + config.WEIGHT_PERSISTENCE * persistence_factor
        )
        risk_score = round(min(risk_score, 1.0), 3)

        # 11. Classify
        risk_level = self._classify(
            risk_score, duration_in_high_state, surge_flag
        )

        return RiskAssessment(
            current_count             = raw_count,
            smoothed_count            = round(self._smoothed_count, 1),
            zone_name                 = config.ZONE_NAME,
            zone_max_capacity         = zone_max,
            zone_safe_capacity        = zone_safe,
            zone_area_m2              = zone_area,
            occupancy_ratio           = round(occupancy_ratio, 3),
            occupancy_pct             = occupancy_pct,
            safe_slots_left           = int(safe_slots_left),
            density_per_m2            = density_per_m2,
            fruin_level               = fruin_level,
            growth_rate               = round(growth_rate, 1),
            fill_rate_per_min         = round(fill_rate_per_min, 2),
            time_to_safe_breach_sec   = t_safe,
            time_to_max_breach_sec    = t_max,
            time_to_safe_breach_label = _format_breach(t_safe),
            time_to_max_breach_label  = _format_breach(t_max),
            risk_score                = risk_score,
            risk_level                = risk_level,
            surge_flag                = surge_flag,
            duration_in_high_state    = duration_in_high_state,
            timestamp                 = now,
        )

    # ── private helpers ───────────────────────────────────

    def _compute_fill_rate(self) -> float:
        """
        Estimate people entering per minute using a linear slope
        over the recent history window.
        Returns 0.0 if crowd is stable or leaving.
        """
        window  = config.BREACH_RATE_WINDOW
        samples = list(self._history)[-window:]
        if len(samples) < 2:
            return 0.0

        t0, c0 = samples[0]
        t1, c1 = samples[-1]
        elapsed_min = (t1 - t0) / 60.0
        if elapsed_min < 0.001:
            return 0.0

        rate = (c1 - c0) / elapsed_min
        return max(rate, 0.0)  # only count people entering, not leaving

    def _compute_breach_times(
        self,
        current: int,
        fill_rate_per_min: float,
        zone_safe: int,
        zone_max: int,
    ) -> tuple:
        """
        Returns (time_to_safe_breach_sec, time_to_max_breach_sec).
        None means stable (not filling) or already breached.
        """
        if fill_rate_per_min <= 0:
            return None, None

        def seconds_to(target: int):
            slots = target - current
            if slots <= 0:
                return None  # already at or past this threshold
            return round((slots / fill_rate_per_min) * 60)

        return seconds_to(zone_safe), seconds_to(zone_max)

    @staticmethod
    def _fruin_classify(density_per_m2: float) -> str:
        if density_per_m2 < config.FRUIN_FREE:
            return "Free"
        elif density_per_m2 < config.FRUIN_RESTRICTED:
            return "Restricted"
        elif density_per_m2 < config.FRUIN_CONSTRAINED:
            return "Constrained"
        elif density_per_m2 < config.FRUIN_DANGEROUS:
            return "Dangerous"
        else:
            return "Critical"

    @staticmethod
    def _classify(
        risk_score: float,
        duration_in_high_state: float,
        surge_flag: bool,
    ) -> str:
        if (
            risk_score >= config.RISK_THRESHOLD_HIGH
            or duration_in_high_state >= config.CRITICAL_PERSISTENCE_SEC
        ):
            level = "Critical"
        elif risk_score >= config.RISK_THRESHOLD_MEDIUM:
            level = "High"
        elif risk_score >= config.RISK_THRESHOLD_LOW:
            level = "Medium"
        else:
            level = "Low"

        if surge_flag and level != "Critical":
            level = "High"

        return level


# ── standalone test ───────────────────────────────────────
# Run: python risk_engine.py
# Verifies logic without needing a camera.

# ── standalone helper ─────────────────────────────────────

def _format_breach(seconds) -> str:
    """Convert seconds to human label e.g. '4 min 20 sec' or 'Stable'."""
    if seconds is None:
        return "Stable"
    if seconds <= 0:
        return "Already breached"
    m, s = divmod(int(seconds), 60)
    if m == 0:
        return f"{s} sec"
    return f"{m} min {s} sec"


# ── standalone test ───────────────────────────────────────
# Run: python risk_engine.py

if __name__ == "__main__":
    import time as _time

    engine = RiskEngine()
    print("--- Simulating crowd filling up ---\n")
    print(
        f"{'Count':>6} | {'Occupancy':>10} | {'Density':>10} | "
        f"{'Fruin':<12} | {'Slots left':>10} | "
        f"{'Time→Safe':>12} | {'Risk':<10}"
    )
    print("-" * 90)

    for count in [5, 10, 18, 25, 32, 40, 48, 55, 65, 78]:
        _time.sleep(0.05)
        r = engine.evaluate(count)
        d = r.to_dict()
        print(
            f"{count:>6} | "
            f"{d['occupancy_pct']:>10} | "
            f"{d['density_per_m2']:>8.3f} p/m² | "
            f"{d['fruin_level']:<12} | "
            f"{d['safe_slots_left']:>10} | "
            f"{str(d['time_to_safe_breach_label']):>12} | "
            f"{d['risk_level']:<10}"
        )