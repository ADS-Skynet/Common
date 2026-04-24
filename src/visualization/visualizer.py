"""
Visualization Tools
Provides visualization utilities for lane detection and LKAS feedback.

Drawing primitives live here so they can be reused by any consumer
(viewer, debug tools, playback scripts). All methods accept plain
numpy arrays and plain-data parameters — no viewer state coupling.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional
from common.types.models import Lane, LaneDepartureStatus


class LKASVisualizer:
    """Visualizer for Lane Keeping Assist System."""

    def __init__(self, image_width: int = 800, image_height: int = 600):
        """
        Initialize visualizer.

        Args:
            image_width: Width of output images
            image_height: Height of output images
        """
        self.image_width = image_width
        self.image_height = image_height

        # Colors (BGR format)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)

    def draw_lanes(
        self,
        image: np.ndarray,
        left_lane: Lane | Tuple[int, int, int, int] | None,
        right_lane: Lane | Tuple[int, int, int, int] | None,
        fill_lane: bool = True,
    ) -> np.ndarray:
        """
        Draw detected lane lines on image.

        Args:
            image: Input image
            left_lane: Left lane (Lane object or tuple (x1, y1, x2, y2))
            right_lane: Right lane (Lane object or tuple (x1, y1, x2, y2))
            fill_lane: Whether to fill the lane area

        Returns:
            Image with drawn lanes
        """
        output = image.copy()

        # Convert Lane objects to tuples for easier access
        def to_tuple(lane):
            if lane is None:
                return None
            if isinstance(lane, Lane):
                return (lane.x1, lane.y1, lane.x2, lane.y2)
            return lane

        left_tuple = to_tuple(left_lane)
        right_tuple = to_tuple(right_lane)

        if fill_lane and left_tuple and right_tuple:
            # Fill lane area
            lane_poly = np.array(
                [
                    [
                        [left_tuple[0], left_tuple[1]],
                        [left_tuple[2], left_tuple[3]],
                        [right_tuple[2], right_tuple[3]],
                        [right_tuple[0], right_tuple[1]],
                    ]
                ],
                dtype=np.int32,
            )

            overlay = output.copy()
            cv2.fillPoly(overlay, lane_poly, self.COLOR_GREEN)
            output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)

        # Draw lane lines
        if left_tuple:
            cv2.line(
                output,
                (left_tuple[0], left_tuple[1]),
                (left_tuple[2], left_tuple[3]),
                self.COLOR_BLUE,
                3,
            )

        if right_tuple:
            cv2.line(
                output,
                (right_tuple[0], right_tuple[1]),
                (right_tuple[2], right_tuple[3]),
                self.COLOR_BLUE,
                3,
            )

        return output

    def draw_segmentation(
        self,
        image: np.ndarray,
        # output: np.ndarray,
        # lanes: list,
        segmentation_mask: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Draw segmentation mask overlay on image.

        Args:
            image: Input image (H, W, 3) BGR or RGB
            segmentation_mask: Segmentation mask (H, W) with class indices
            alpha: Blend factor for overlay (0-1)

        Returns:
            Image with segmentation overlay
        """
        output = image.copy()

        # Ensure mask matches image dimensions
        if segmentation_mask.shape[:2] != image.shape[:2]:
            segmentation_mask = cv2.resize(
                segmentation_mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Create colored overlay for lane pixels (class > 0)
        overlay = np.zeros_like(output)
        lane_pixels = segmentation_mask > 0

        # Transparent blue color for lane (RGB format: blue with some green for visibility)
        overlay[lane_pixels] = [250, 100, 50]  # Light blue in RGB

        # Blend overlay with original image
        output = cv2.addWeighted(output, 1 - alpha, overlay, alpha, 0)

        return output

    def draw_vehicle_position(
        self,
        image: np.ndarray,
        vehicle_center_x: int,
        lane_center_x: float | None,
        departure_status: LaneDepartureStatus,
    ) -> np.ndarray:
        """
        Draw vehicle position indicator.

        Args:
            image: Input image
            vehicle_center_x: X coordinate of vehicle center
            lane_center_x: X coordinate of lane center
            departure_status: Current departure status

        Returns:
            Image with position indicator
        """
        output = image.copy()
        height = output.shape[0]

        # Draw vehicle center line (convert to int for OpenCV)
        cv2.line(
            output,
            (int(vehicle_center_x), height - 50),
            (int(vehicle_center_x), height),
            self.COLOR_WHITE,
            2,
        )

        # Draw lane center line if available
        if lane_center_x is not None:
            color = self._get_status_color(departure_status)
            cv2.line(
                output,
                (int(lane_center_x), height - 50),
                (int(lane_center_x), height),
                color,
                2,
            )

            # Draw offset arrow
            if abs(vehicle_center_x - lane_center_x) > 5:
                cv2.arrowedLine(
                    output,
                    (int(vehicle_center_x), height - 25),
                    (int(lane_center_x), height - 25),
                    color,
                    2,
                    tipLength=0.3,
                )

        return output

    def draw_hud(
        self,
        image: np.ndarray,
        metrics: Dict,
        show_steering: bool = True,
        steering_value: float | None = None,
        vehicle_telemetry: Dict | None = None,
    ) -> np.ndarray:
        """
        Draw heads-up display with metrics.

        Args:
            image: Input image
            metrics: Dictionary of lane metrics
            show_steering: Whether to show steering indicator
            steering_value: Steering correction value [-1, 1]
            vehicle_telemetry: Optional dict with vehicle data (speed, position, etc.)

        Returns:
            Image with HUD overlay
        """
        output = image.copy()

        # Create semi-transparent overlay for HUD
        overlay = output.copy()
        hud_height = 200 if vehicle_telemetry else 150
        cv2.rectangle(
            overlay, (0, 0), (output.shape[1], hud_height), self.COLOR_BLACK, -1
        )
        output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)

        # Display metrics
        y_offset = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Departure status
        status = metrics.get("departure_status", LaneDepartureStatus.NO_LANES)
        status_color = self._get_status_color(status)
        cv2.putText(
            output,
            f"Status: {status.value.upper()}",
            (10, y_offset),
            font,
            font_scale,
            status_color,
            thickness,
        )
        y_offset += 30

        # Lateral offset
        offset_m = metrics.get("lateral_offset_meters")
        if offset_m is not None:
            direction = "RIGHT" if offset_m > 0 else "LEFT"
            cv2.putText(
                output,
                f"Offset: {abs(offset_m):.2f}m {direction}",
                (10, y_offset),
                font,
                font_scale,
                self.COLOR_WHITE,
                thickness,
            )
        else:
            cv2.putText(
                output,
                "Offset: N/A",
                (10, y_offset),
                font,
                font_scale,
                self.COLOR_WHITE,
                thickness,
            )
        y_offset += 30

        # Heading angle
        heading = metrics.get("heading_angle_deg")
        if heading is not None:
            cv2.putText(
                output,
                f"Heading: {heading:.1f} deg",
                (10, y_offset),
                font,
                font_scale,
                self.COLOR_WHITE,
                thickness,
            )
        else:
            cv2.putText(
                output,
                "Heading: N/A",
                (10, y_offset),
                font,
                font_scale,
                self.COLOR_WHITE,
                thickness,
            )
        y_offset += 30

        # Lane width
        lane_width = metrics.get("lane_width_pixels")
        if lane_width is not None:
            cv2.putText(
                output,
                f"Lane Width: {lane_width:.0f}px",
                (10, y_offset),
                font,
                font_scale,
                self.COLOR_WHITE,
                thickness,
            )

        # Vehicle telemetry (if provided)
        if vehicle_telemetry:
            y_offset += 40

            # Speed and Throttle
            speed = vehicle_telemetry.get("speed_kmh", 0)
            throttle = vehicle_telemetry.get("throttle", 0)
            cv2.putText(
                output,
                f"Speed: {speed:.1f} km/h | Throttle: {throttle:.2f}",
                (10, y_offset),
                font,
                font_scale,
                self.COLOR_GREEN,
                thickness,
            )

            # Position
            pos = vehicle_telemetry.get("position")
            if pos:
                y_offset += 30
                cv2.putText(
                    output,
                    f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                    (10, y_offset),
                    font,
                    0.5,
                    self.COLOR_WHITE,
                    1,
                )

        # Steering indicator
        if show_steering and steering_value is not None:
            self._draw_steering_indicator(output, steering_value)

        return output

    def _draw_steering_indicator(self, image: np.ndarray, steering_value: float):
        """
        Draw steering wheel indicator.

        Args:
            image: Input image
            steering_value: Steering value [-1, 1]
        """
        # Position in top-right corner
        center_x = image.shape[1] - 100
        center_y = 75
        radius = 50

        # Draw steering wheel circle
        cv2.circle(image, (center_x, center_y), radius, self.COLOR_WHITE, 2)

        # Draw center mark
        cv2.circle(image, (center_x, center_y), 3, self.COLOR_WHITE, -1)

        # Draw steering indicator
        angle = steering_value * 90  # Max ±90 degrees
        angle_rad = np.radians(angle - 90)  # Adjust for coordinate system

        end_x = int(center_x + radius * 0.8 * np.cos(angle_rad))
        end_y = int(center_y + radius * 0.8 * np.sin(angle_rad))

        color = (
            self.COLOR_GREEN
            if abs(steering_value) < 0.3
            else self.COLOR_YELLOW if abs(steering_value) < 0.6 else self.COLOR_RED
        )

        cv2.line(image, (center_x, center_y), (end_x, end_y), color, 3)

        # Label
        cv2.putText(
            image,
            f"{steering_value:.2f}",
            (center_x - 30, center_y + radius + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.COLOR_WHITE,
            2,
        )

    def _get_status_color(self, status: LaneDepartureStatus) -> Tuple[int, int, int]:
        """
        Get color based on departure status.

        Args:
            status: Departure status

        Returns:
            BGR color tuple
        """
        if status == LaneDepartureStatus.CENTERED:
            return self.COLOR_GREEN
        elif status in [
            LaneDepartureStatus.LEFT_DRIFT,
            LaneDepartureStatus.RIGHT_DRIFT,
        ]:
            return self.COLOR_YELLOW
        elif status in [
            LaneDepartureStatus.LEFT_DEPARTURE,
            LaneDepartureStatus.RIGHT_DEPARTURE,
        ]:
            return self.COLOR_RED
        else:
            return self.COLOR_WHITE

    def draw_canny_edges(
        self,
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150,
    ) -> np.ndarray:
        """
        Produce a Canny edge visualization (white edges on black background).

        Args:
            image: Input RGB image
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold

        Returns:
            RGB image with white edges on black
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, low_threshold, high_threshold)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    def draw_hough_lines(
        self,
        image: np.ndarray,
        roi_config: Dict[str, float],
        canny_low: int = 50,
        canny_high: int = 150,
        hough_rho: int = 2,
        hough_theta: float = 0.017453,
        hough_threshold: int = 50,
        min_line_len: int = 40,
        max_line_gap: int = 100,
    ) -> np.ndarray:
        """
        Produce a Hough lines visualization (magenta lines on black background).

        Applies Canny + ROI masking + HoughLinesP, then draws the resulting
        line segments.

        Args:
            image: Input RGB image
            roi_config: ROI trapezoid fractions
                (roi_bottom_left_x, roi_top_left_x, roi_top_right_x,
                 roi_bottom_right_x, roi_top_y)
            canny_low: Canny low threshold
            canny_high: Canny high threshold
            hough_rho: Accumulator distance resolution (pixels)
            hough_theta: Accumulator angle resolution (radians)
            hough_threshold: Voting threshold
            min_line_len: Minimum line length (pixels)
            max_line_gap: Maximum gap between segments (pixels)

        Returns:
            RGB image with magenta lines on black
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, canny_low, canny_high)

        height, width = edges.shape
        mask = np.zeros_like(edges)
        vertices = np.array([[
            [int(width * roi_config['roi_bottom_left_x']), height],
            [int(width * roi_config['roi_top_left_x']), int(height * roi_config['roi_top_y'])],
            [int(width * roi_config['roi_top_right_x']), int(height * roi_config['roi_top_y'])],
            [int(width * roi_config['roi_bottom_right_x']), height],
        ]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            masked_edges, hough_rho, hough_theta, hough_threshold,
            minLineLength=min_line_len, maxLineGap=max_line_gap,
        )

        output = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output, (x1, y1), (x2, y2), (255, 0, 255), 2)

        return output

    def draw_roi(
        self,
        image: np.ndarray,
        roi_config: Dict[str, float],
    ) -> np.ndarray:
        """
        Draw the ROI trapezoid outline (green) used by CV detection.

        Args:
            image: Input image (modified in-place)
            roi_config: ROI trapezoid fractions

        Returns:
            Same image with ROI polyline drawn
        """
        height, width = image.shape[:2]
        vertices = np.array([[
            [int(width * roi_config['roi_bottom_left_x']), height],
            [int(width * roi_config['roi_top_left_x']), int(height * roi_config['roi_top_y'])],
            [int(width * roi_config['roi_top_right_x']), int(height * roi_config['roi_top_y'])],
            [int(width * roi_config['roi_bottom_right_x']), height],
        ]], dtype=np.int32)
        cv2.polylines(image, vertices, True, self.COLOR_GREEN, 2)
        return image

    def draw_polynomials(
        self,
        image: np.ndarray,
        left_poly: Optional[List[float]] = None,
        right_poly: Optional[List[float]] = None,
        center_poly: Optional[List[float]] = None,
        left_confidence: float = 0.0,
        right_confidence: float = 0.0,
        camera_offset_x: int = 0,
    ) -> np.ndarray:
        """
        Draw polynomial lane boundary curves and center path.

        Each polynomial is ``x = a*y^2 + b*y + c`` where coefficients are
        ``[a, b, c]``.

        Colors: left = blue-ish, right = red-ish, center = yellow/cyan.

        Args:
            image: Input image (modified in-place)
            left_poly: Left boundary coefficients [a, b, c] or None
            right_poly: Right boundary coefficients [a, b, c] or None
            center_poly: Center path coefficients [a, b, c] or None
            left_confidence: Confidence score [0, 1] for left boundary
            right_confidence: Confidence score [0, 1] for right boundary
            camera_offset_x: Camera offset from vehicle center (pixels)

        Returns:
            Same image with polynomial curves drawn
        """
        height, width = image.shape[:2]
        y_end = int(height * 0.4)
        y_values = np.arange(height - 1, y_end, -2)

        def _eval(coeffs, y_vals):
            a, b, c = coeffs
            return a * y_vals ** 2 + b * y_vals + c

        def _draw_curve(coeffs, color, thickness=2):
            x_vals = _eval(coeffs, y_values.astype(np.float64))
            valid = (x_vals >= 0) & (x_vals < width)
            if not np.any(valid):
                return
            pts = np.column_stack([
                x_vals[valid].astype(np.int32),
                y_values[valid].astype(np.int32),
            ])
            if len(pts) >= 2:
                cv2.polylines(image, [pts], isClosed=False,
                              color=color, thickness=thickness)

        # Left boundary
        if left_poly:
            _draw_curve(left_poly, color=(255, 100, 100), thickness=2)
        if left_confidence > 0:
            lx = int(_eval(left_poly, float(y_end + 20))) if left_poly else 40
            lx = max(5, min(width - 80, lx))
            cv2.putText(image, f"L:{left_confidence:.2f}", (lx, y_end + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)

        # Right boundary
        if right_poly:
            _draw_curve(right_poly, color=(100, 100, 255), thickness=2)
        if right_confidence > 0:
            rx = int(_eval(right_poly, float(y_end + 20))) if right_poly else width - 80
            rx = max(5, min(width - 80, rx))
            cv2.putText(image, f"R:{right_confidence:.2f}", (rx, y_end + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

        # Center path
        if center_poly:
            _draw_curve(center_poly, color=(0, 255, 255), thickness=3)

            # Lookahead marker
            lookahead_y = int(height * 0.6)
            lookahead_x = _eval(center_poly, float(lookahead_y))
            if 0 <= lookahead_x < width:
                cv2.circle(image, (int(lookahead_x), lookahead_y),
                           6, (0, 255, 255), -1)

            # Vehicle center reference line
            cx = width // 2 + camera_offset_x
            cv2.line(image, (cx, height - 1), (cx, y_end),
                     self.COLOR_WHITE, 1)

        return image

    def create_alert_overlay(
        self,
        image: np.ndarray,
        departure_status: LaneDepartureStatus,
        blink: bool = False,
    ) -> np.ndarray:
        """
        Create visual alert overlay for lane departure warnings.

        Args:
            image: Input image
            departure_status: Current departure status
            blink: Whether to show blinking effect

        Returns:
            Image with alert overlay
        """
        output = image.copy()

        if departure_status in [
            LaneDepartureStatus.LEFT_DEPARTURE,
            LaneDepartureStatus.RIGHT_DEPARTURE,
        ]:
            if not blink or (blink and np.random.rand() > 0.5):
                # Create red border
                border_thickness = 10
                color = self.COLOR_RED

                height, width = output.shape[:2]
                cv2.rectangle(output, (0, 0), (width, height), color, border_thickness)

                # Warning text
                warning_text = "LANE DEPARTURE WARNING!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3

                text_size = cv2.getTextSize(warning_text, font, font_scale, thickness)[
                    0
                ]
                text_x = (width - text_size[0]) // 2
                text_y = 50

                # Draw text with background
                cv2.rectangle(
                    output,
                    (text_x - 10, text_y - text_size[1] - 10),
                    (text_x + text_size[0] + 10, text_y + 10),
                    self.COLOR_BLACK,
                    -1,
                )
                cv2.putText(
                    output,
                    warning_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    color,
                    thickness,
                )

        elif departure_status in [
            LaneDepartureStatus.LEFT_DRIFT,
            LaneDepartureStatus.RIGHT_DRIFT,
        ]:
            # Yellow border for drift warning
            border_thickness = 5
            color = self.COLOR_YELLOW

            height, width = output.shape[:2]
            cv2.rectangle(output, (0, 0), (width, height), color, border_thickness)

        return output

    def create_combined_view(
        self,
        original_image: np.ndarray,
        processed_image: np.ndarray,
        metrics: Dict,
        steering_value: float | None = None,
    ) -> np.ndarray:
        """
        Create combined view with original and processed images side by side.

        Args:
            original_image: Original camera image
            processed_image: Processed image with lane detection
            metrics: Lane metrics dictionary
            steering_value: Steering correction value

        Returns:
            Combined visualization
        """
        # Resize images to same size
        height = 480
        aspect_ratio = original_image.shape[1] / original_image.shape[0]
        width = int(height * aspect_ratio)

        original_resized = cv2.resize(original_image, (width, height))
        processed_resized = cv2.resize(processed_image, (width, height))

        # Create combined image
        combined = np.hstack([original_resized, processed_resized])

        # Add HUD to combined view
        combined = self.draw_hud(
            combined, metrics, show_steering=True, steering_value=steering_value
        )

        # Add labels
        cv2.putText(
            combined,
            "ORIGINAL",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.COLOR_WHITE,
            2,
        )
        cv2.putText(
            combined,
            "LANE DETECTION",
            (width + 10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.COLOR_WHITE,
            2,
        )

        return combined
