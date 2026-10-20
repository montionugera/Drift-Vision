"""Road lines and lane markings detection in video frames.

Includes edge detection, ROI masking, line segment extraction, slope-based
lane clustering, and temporal smoothing across video frames.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import cv2
import numpy as np


@dataclass
class LaneLine:
    """A lane line segment defined by two endpoints (x1, y1) and (x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def slope(self) -> float:
        """Slope (m) of the line in image coordinates (y downwards)."""
        dx = self.x2 - self.x1
        return (self.y2 - self.y1) / dx if dx != 0 else float("inf")

    @property
    def intercept(self) -> float:
        """Y-intercept (b) of the line (y = m*x + b)."""
        m = self.slope
        return self.y1 - m * self.x1 if m != float("inf") else 0.0

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return integer coordinate tuple (x1, y1, x2, y2)."""
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        )


LineCoords = tuple[int, int, int, int]
LineInput = Union[LineCoords, tuple[float, float, float, float], list[int], list[float], LaneLine]


def _normalize_line(line: LineInput) -> tuple[float, float, float, float]:
    """Convert line input into a tuple of 4 floats (x1, y1, x2, y2)."""
    if isinstance(line, LaneLine):
        return (float(line.x1), float(line.y1), float(line.x2), float(line.y2))
    if isinstance(line, (tuple, list)) and len(line) == 4:
        return (float(line[0]), float(line[1]), float(line[2]), float(line[3]))
    raise ValueError(f"Invalid line representation: {line}")


class LaneSmoother:
    """Temporal rolling-window smoother for detected lane lines across frames.

    Maintains a FIFO rolling window of the last `window_size` detected lines for
    both the left and right lane boundaries. Averages the line coordinates over
    the window to eliminate inter-frame jitter and maintain steady lines in video.

    If a lane is briefly lost (e.g. due to occlusion or lighting variation for a
    frame), it retains the averaged estimate for up to `max_missing_frames`
    consecutive frames before resetting to None.

    Args:
        window_size: Number of frames to average over (rolling window, default 5).
        max_missing_frames: Max consecutive frames to bridge when a lane is missing.
    """

    def __init__(self, window_size: int = 5, max_missing_frames: int = 3) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative")
        self.window_size = window_size
        self.max_missing_frames = max_missing_frames
        self.left_history: deque[tuple[float, float, float, float]] = deque(maxlen=window_size)
        self.right_history: deque[tuple[float, float, float, float]] = deque(maxlen=window_size)
        self.left_missing_count: int = 0
        self.right_missing_count: int = 0

    def _update_lane(
        self,
        current_line: Optional[LineInput],
        history: deque[tuple[float, float, float, float]],
        is_left: bool,
    ) -> Optional[tuple[int, int, int, int]]:
        missing_count = self.left_missing_count if is_left else self.right_missing_count

        if current_line is not None:
            normalized = _normalize_line(current_line)
            history.append(normalized)
            if is_left:
                self.left_missing_count = 0
            else:
                self.right_missing_count = 0
        else:
            missing_count += 1
            if is_left:
                self.left_missing_count = missing_count
            else:
                self.right_missing_count = missing_count

            if missing_count > self.max_missing_frames:
                history.clear()
                return None

        if not history:
            return None

        avg = np.mean(history, axis=0)
        return (int(round(avg[0])), int(round(avg[1])), int(round(avg[2])), int(round(avg[3])))

    def update(
        self,
        left_line: Optional[LineInput] = None,
        right_line: Optional[LineInput] = None,
    ) -> tuple[Optional[tuple[int, int, int, int]], Optional[tuple[int, int, int, int]]]:
        """Update smoother with detections from the latest frame.

        Returns:
            Tuple of (smoothed_left, smoothed_right) as (x1, y1, x2, y2) tuples or None.
        """
        smoothed_left = self._update_lane(left_line, self.left_history, is_left=True)
        smoothed_right = self._update_lane(right_line, self.right_history, is_left=False)
        return smoothed_left, smoothed_right

    def smooth(
        self,
        left_line: Optional[LineInput] = None,
        right_line: Optional[LineInput] = None,
    ) -> tuple[Optional[tuple[int, int, int, int]], Optional[tuple[int, int, int, int]]]:
        """Alias for update()."""
        return self.update(left_line, right_line)

    @property
    def current_left(self) -> Optional[tuple[int, int, int, int]]:
        """Current smoothed left lane without adding a new frame."""
        if not self.left_history:
            return None
        avg = np.mean(self.left_history, axis=0)
        return (int(round(avg[0])), int(round(avg[1])), int(round(avg[2])), int(round(avg[3])))

    @property
    def current_right(self) -> Optional[tuple[int, int, int, int]]:
        """Current smoothed right lane without adding a new frame."""
        if not self.right_history:
            return None
        avg = np.mean(self.right_history, axis=0)
        return (int(round(avg[0])), int(round(avg[1])), int(round(avg[2])), int(round(avg[3])))

    def reset(self) -> None:
        """Clear all history buffers and counters."""
        self.left_history.clear()
        self.right_history.clear()
        self.left_missing_count = 0
        self.right_missing_count = 0


def detect_edges(
    frame: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_kernel: int = 5,
) -> np.ndarray:
    """Convert frame to grayscale, apply Gaussian blur, and compute Canny edges."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    return cv2.Canny(blurred, canny_low, canny_high)


def region_of_interest(
    edges: np.ndarray,
    roi_top_y: float = 0.6,
    roi_top_width: float = 0.15,
) -> np.ndarray:
    """Mask out everything outside the trapezoidal road area in front of the car."""
    height, width = edges.shape[:2]
    mask = np.zeros_like(edges)

    top_y = int(height * roi_top_y)
    top_w = int(width * roi_top_width)
    center_x = width // 2

    vertices = np.array(
        [
            [
                (int(width * 0.05), height),
                (center_x - top_w // 2, top_y),
                (center_x + top_w // 2, top_y),
                (int(width * 0.95), height),
            ]
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)


def detect_line_segments(
    masked_edges: np.ndarray,
    rho: float = 1.0,
    theta: float = np.pi / 180,
    threshold: int = 20,
    min_line_len: int = 20,
    max_line_gap: int = 5,
) -> list[tuple[int, int, int, int]]:
    """Detect line segments using probabilistic Hough transform."""
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []
    res = []
    for line in lines:
        coords = line[0] if (len(line) == 1 and hasattr(line[0], "__len__")) else line
        x1, y1, x2, y2 = coords
        res.append((int(x1), int(y1), int(x2), int(y2)))
    return res


def split_left_right(
    lines: Sequence[tuple[int, int, int, int]],
    min_slope: float = 0.3,
    max_slope: float = 3.0,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Split line segments by slope into left (negative) and right (positive) lanes.

    Near-horizontal and near-vertical segments outside [min_slope, max_slope] are ignored.
    """
    left_lines = []
    right_lines = []

    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        if dx == 0:
            continue
        slope = (y2 - y1) / dx
        abs_slope = abs(slope)
        if abs_slope < min_slope or abs_slope > max_slope:
            continue

        if slope < 0:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines


def average_lane_segments(
    segments: Sequence[tuple[int, int, int, int]],
    y_bottom: int,
    y_top: int,
) -> Optional[tuple[int, int, int, int]]:
    """Fit a single line to segments and extrapolate between y_bottom and y_top."""
    if not segments:
        return None

    xs = []
    ys = []
    weights = []

    for x1, y1, x2, y2 in segments:
        length = np.hypot(x2 - x1, y2 - y1)
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        weights.extend([length, length])

    if len(xs) < 2:
        return None

    # Fit x = m*y + b (fit x as a function of y to avoid infinite slope issues)
    try:
        poly = np.polyfit(ys, xs, deg=1, w=weights)
    except Exception:
        return None

    x_bottom = int(round(np.polyval(poly, y_bottom)))
    x_top = int(round(np.polyval(poly, y_top)))

    return (x_bottom, y_bottom, x_top, y_top)


def detect_lanes_in_frame(
    frame: np.ndarray,
    config: Optional[dict[str, Any]] = None,
    smoother: Optional[LaneSmoother] = None,
) -> tuple[Optional[tuple[int, int, int, int]], Optional[tuple[int, int, int, int]]]:
    """Run full lane detection pipeline on a single frame, with optional temporal smoothing."""
    cfg = config or {}
    canny_low = cfg.get("canny_low", 50)
    canny_high = cfg.get("canny_high", 150)
    roi_top_y = cfg.get("roi_top_y", 0.6)
    roi_top_width = cfg.get("roi_top_width", 0.15)

    edges = detect_edges(frame, canny_low=canny_low, canny_high=canny_high)
    roi = region_of_interest(edges, roi_top_y=roi_top_y, roi_top_width=roi_top_width)
    segments = detect_line_segments(roi)
    left_segs, right_segs = split_left_right(segments)

    height = frame.shape[0]
    y_bottom = height
    y_top = int(height * roi_top_y)

    raw_left = average_lane_segments(left_segs, y_bottom=y_bottom, y_top=y_top)
    raw_right = average_lane_segments(right_segs, y_bottom=y_bottom, y_top=y_top)

    if smoother is not None:
        return smoother.update(raw_left, raw_right)
    return raw_left, raw_right
