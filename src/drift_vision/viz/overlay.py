"""Visualization overlay: drawing lane lines, detected objects, and turn information."""

from typing import Optional, Union

import cv2
import numpy as np

from drift_vision.perception.lanes import LaneLine

LineType = Optional[Union[tuple[int, int, int, int], LaneLine]]


def _extract_coords(line: LineType) -> Optional[tuple[int, int, int, int]]:
    if line is None:
        return None
    if isinstance(line, LaneLine):
        return line.as_tuple()
    if isinstance(line, (tuple, list)) and len(line) == 4:
        return (int(round(line[0])), int(round(line[1])), int(round(line[2])), int(round(line[3])))
    return None


def draw_lane_lines(
    frame: np.ndarray,
    left_line: LineType = None,
    right_line: LineType = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 5,
) -> np.ndarray:
    """Draw left and right lane lines onto a copy of the frame."""
    out = frame.copy()

    left_coords = _extract_coords(left_line)
    if left_coords is not None:
        x1, y1, x2, y2 = left_coords
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)

    right_coords = _extract_coords(right_line)
    if right_coords is not None:
        x1, y1, x2, y2 = right_coords
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)

    return out


def draw_lane_area(
    frame: np.ndarray,
    left_line: LineType = None,
    right_line: LineType = None,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.3,
) -> np.ndarray:
    """Fill the polygon enclosed between left and right lane lines with a semi-transparent color."""
    left_coords = _extract_coords(left_line)
    right_coords = _extract_coords(right_line)

    if left_coords is None or right_coords is None:
        return frame.copy()

    lx1, ly1, lx2, ly2 = left_coords
    rx1, ry1, rx2, ry2 = right_coords

    pts = np.array([[(lx1, ly1), (lx2, ly2), (rx2, ry2), (rx1, ry1)]], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, pts, color)

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
