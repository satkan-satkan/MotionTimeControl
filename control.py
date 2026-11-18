"""Camera-independent gesture mapping, calibration and jitter suppression."""
from dataclasses import dataclass
import math


def pinch_ratio(landmarks, width: int, height: int) -> float | None:
    """Thumb/index distance divided by wrist-to-middle-knuckle distance.

    Use pixel distances: normalized x/y use different units on non-square frames.
    Palm normalization reduces sensitivity to distance from the camera.
    """
    if len(landmarks) < 21:
        return None

    def distance(a, b):
        return math.hypot((landmarks[a].x - landmarks[b].x) * width,
                          (landmarks[a].y - landmarks[b].y) * height)

    palm = distance(0, 9)
    if not math.isfinite(palm) or palm < 12:
        return None
    ratio = distance(4, 8) / palm
    return ratio if math.isfinite(ratio) else None


@dataclass
class TimelineControl:
    closed: float = 0.18
    opened: float = 1.45
    position: float = 0.0
    response_seconds: float = 0.09
    deadband: float = 0.004

    def calibrate(self, closed: float, opened: float):
        if not all(math.isfinite(v) for v in (closed, opened)) or closed < 0:
            raise ValueError('Invalid calibration measurement.')
        if opened - closed < 0.25:
            raise ValueError('Open your fingers wider, then press C again.')
        self.closed, self.opened = closed, opened

    def update(self, ratio: float | None, dt: float, frozen=False) -> float:
        # Losing a hand holds the exact current frame, including smoothing state.
        if ratio is None or frozen or not math.isfinite(ratio):
            return self.position
        target = max(0.0, min(1.0, (ratio - self.closed) / (self.opened - self.closed)))
        difference = target - self.position
        if abs(difference) > self.deadband:
            alpha = -math.expm1(-min(max(dt, 0), 0.1) / self.response_seconds)
            self.position += alpha * difference
        # Endpoints should reach the first and last frames exactly.
        if target in (0.0, 1.0) and abs(target - self.position) < self.deadband:
            self.position = target
        return self.position

    def seek(self, position):
        self.position = max(0.0, min(1.0, position))
