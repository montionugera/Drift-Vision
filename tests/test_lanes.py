import cv2
import numpy as np
import pytest

from drift_vision.perception.lanes import (
    LaneLine,
    LaneSmoother,
    average_lane_segments,
    detect_edges,
    detect_lanes_in_frame,
    detect_line_segments,
    region_of_interest,
    split_left_right,
)
from drift_vision.viz.overlay import draw_lane_area, draw_lane_lines


class TestLaneLine:
    def test_properties_and_tuple_conversion(self):
        line = LaneLine(100.0, 500.0, 200.0, 300.0)
        # slope = (300 - 500) / (200 - 100) = -2.0
        assert line.slope == pytest.approx(-2.0)
        # intercept = 500 - (-2.0 * 100) = 700.0
        assert line.intercept == pytest.approx(700.0)
        assert line.as_tuple() == (100, 500, 200, 300)

    def test_vertical_line_slope(self):
        line = LaneLine(100.0, 500.0, 100.0, 300.0)
        assert line.slope == float("inf")
        assert line.intercept == 0.0


class TestLaneSmoother:
    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="window_size must be at least 1"):
            LaneSmoother(window_size=0)
        with pytest.raises(ValueError, match="max_missing_frames must be non-negative"):
            LaneSmoother(max_missing_frames=-1)

    def test_single_frame_returns_exact_coordinates(self):
        smoother = LaneSmoother(window_size=5)
        left = (200, 720, 500, 450)
        right = (1080, 720, 780, 450)
        s_left, s_right = smoother.update(left, right)
        assert s_left == left
        assert s_right == right

    def test_averages_over_rolling_window_to_remove_jitter(self):
        smoother = LaneSmoother(window_size=3)

        # Jittery input over 3 frames
        f1_left = (200, 720, 500, 450)
        f2_left = (210, 720, 510, 450)
        f3_left = (196, 720, 496, 450)

        s1_left, _ = smoother.update(left_line=f1_left)
        assert s1_left == f1_left

        s2_left, _ = smoother.update(left_line=f2_left)
        # Average of (200, 210) = 205; (500, 510) = 505
        assert s2_left == (205, 720, 505, 450)

        s3_left, _ = smoother.update(left_line=f3_left)
        # Average of (200 + 210 + 196) / 3 = 202; (500 + 510 + 496) / 3 = 502
        assert s3_left == (202, 720, 502, 450)

    def test_rolling_window_drops_old_frames(self):
        smoother = LaneSmoother(window_size=2)
        smoother.update(left_line=(100, 500, 200, 300))
        smoother.update(left_line=(200, 500, 300, 300))

        # Frame 3 drops Frame 1 (100 -> 200, 300)
        s_left, _ = smoother.update(left_line=(300, 500, 400, 300))
        # Average of frame 2 and 3: (200+300)/2 = 250, (300+400)/2 = 350
        assert s_left == (250, 500, 350, 300)

    def test_missing_frame_bridging_and_reset(self):
        smoother = LaneSmoother(window_size=3, max_missing_frames=2)
        left = (200, 720, 500, 450)
        smoother.update(left_line=left)

        # 1st missing frame: still uses history
        s_left_1, _ = smoother.update(left_line=None)
        assert s_left_1 == left

        # 2nd missing frame: still uses history
        s_left_2, _ = smoother.update(left_line=None)
        assert s_left_2 == left

        # 3rd missing frame: exceeds max_missing_frames (2), resets to None
        s_left_3, _ = smoother.update(left_line=None)
        assert s_left_3 is None

    def test_reset_clears_history(self):
        smoother = LaneSmoother(window_size=5)
        smoother.update(left_line=(200, 720, 500, 450), right_line=(1080, 720, 780, 450))
        assert smoother.current_left is not None
        assert smoother.current_right is not None

        smoother.reset()
        assert smoother.current_left is None
        assert smoother.current_right is None

    def test_accepts_lane_line_objects_and_lists(self):
        smoother = LaneSmoother(window_size=3)
        l_obj = LaneLine(200, 700, 400, 400)
        r_list = [800, 700, 600, 400]
        s_left, s_right = smoother.update(l_obj, r_list)
        assert s_left == (200, 700, 400, 400)
        assert s_right == (800, 700, 600, 400)


    def test_smoothing_reduces_variance_of_jittery_signal(self):
        # Simulate 20 frames of a lane line with Gaussian noise/jitter
        np.random.seed(42)
        true_x1 = 200.0
        true_x2 = 500.0
        y1, y2 = 720, 450

        raw_x1s = []
        smoothed_x1s = []

        smoother = LaneSmoother(window_size=5)

        for _ in range(30):
            noisy_x1 = true_x1 + np.random.normal(0, 15.0)
            noisy_x2 = true_x2 + np.random.normal(0, 10.0)
            raw_line = (noisy_x1, y1, noisy_x2, y2)
            raw_x1s.append(noisy_x1)

            smoothed_left, _ = smoother.update(left_line=raw_line)
            smoothed_x1s.append(smoothed_left[0])

        # Ignore initial burn-in window (first 5 frames)
        raw_var = np.var(raw_x1s[5:])
        smoothed_var = np.var(smoothed_x1s[5:])

        # Smoothed variance should be significantly lower than raw jitter variance
        assert smoothed_var < raw_var * 0.5


class TestLaneDetectionPipeline:
    def test_synthetic_frame_detection_and_smoothing(self):
        # Create a synthetic image with two white lane lines on black background
        h, w = 720, 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Left lane line: negative slope (200, 720) to (550, 440)
        cv2.line(frame, (200, 720), (550, 440), (255, 255, 255), thickness=10)
        # Right lane line: positive slope (1080, 720) to (730, 440)
        cv2.line(frame, (1080, 720), (730, 440), (255, 255, 255), thickness=10)

        smoother = LaneSmoother(window_size=3)
        left, right = detect_lanes_in_frame(frame, smoother=smoother)

        assert left is not None
        assert right is not None

        # Left x should be left of center, right x right of center
        lx1, ly1, lx2, ly2 = left
        rx1, ry1, rx2, ry2 = right
        assert lx1 < w // 2
        assert rx1 > w // 2
        assert lx2 < rx2

    def test_overlay_drawing(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        left = (200, 720, 500, 450)
        right = (1080, 720, 780, 450)

        out_lines = draw_lane_lines(frame, left, right)
        assert out_lines.shape == frame.shape
        assert np.any(out_lines > 0)

        out_area = draw_lane_area(frame, left, right)
        assert out_area.shape == frame.shape
        assert np.any(out_area > 0)
