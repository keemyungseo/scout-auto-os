"""Expected vs actual progress tracking."""

from __future__ import annotations

from dataclasses import dataclass

from scout_auto_os.engine.expectation.curve_builder import ExpectedPath


@dataclass
class ProgressSnapshot:
    current_elapsed: int
    expected_roi_now: float
    current_roi: float
    expected_progress: float
    progress_delta: float
    progress_ratio: float
    time_progress_ratio: float


def interpolate_expected_roi(path: ExpectedPath, elapsed_min: int) -> float:
    curve = path.expected_progress_curve
    if not curve:
        return 0.0
    if elapsed_min <= 0:
        return 0.0
    pts = sorted(curve, key=lambda p: p["minute"])
    if elapsed_min >= pts[-1]["minute"]:
        return float(pts[-1]["roi_pct"])
    for i in range(1, len(pts)):
        lo, hi = pts[i - 1], pts[i]
        if lo["minute"] <= elapsed_min <= hi["minute"]:
            span = hi["minute"] - lo["minute"]
            if span <= 0:
                return float(hi["roi_pct"])
            w = (elapsed_min - lo["minute"]) / span
            return round(float(lo["roi_pct"]) + w * (float(hi["roi_pct"]) - float(lo["roi_pct"])), 4)
    return float(pts[-1]["roi_pct"])


def compute_progress(path: ExpectedPath, elapsed_min: int, current_roi: float) -> ProgressSnapshot:
    exp_now = interpolate_expected_roi(path, elapsed_min)
    time_ratio = elapsed_min / max(path.expected_horizon, 1) * 100
    if abs(exp_now) < 0.01:
        ratio = 100.0 if current_roi >= 0 else 0.0
    else:
        ratio = round(current_roi / exp_now * 100, 2)
    exp_prog = round(min(100.0, time_ratio), 2)
    delta = round(current_roi - exp_now, 4)
    return ProgressSnapshot(
        current_elapsed=elapsed_min,
        expected_roi_now=exp_now,
        current_roi=round(current_roi, 4),
        expected_progress=exp_prog,
        progress_delta=delta,
        progress_ratio=ratio,
        time_progress_ratio=round(time_ratio, 2),
    )
