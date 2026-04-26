"""
Data models for lane detection system.
Uses dataclasses for type safety and clean data structures.
"""

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class LaneDepartureStatus(Enum):
    """Lane departure status enumeration."""
    UNKNOWN = "Unknown"
    CENTERED = "Centered"
    LEFT_DRIFT = "Drifting Left"
    RIGHT_DRIFT = "Drifting Right"
    LEFT_DEPARTURE = "LEFT DEPARTURE!"
    RIGHT_DEPARTURE = "RIGHT DEPARTURE!"
    NO_LANES = "No Lanes Detected"


@dataclass
class Lane:
    """
    Represents a detected lane line.

    Attributes:
        x1, y1: Starting point (bottom of image)
        x2, y2: Ending point (top of ROI)
        confidence: Detection confidence [0, 1]
    """
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0


@dataclass
class LaneContour:
    """
    Lane contour representation for DL detection (multiple points).

    Attributes:
        points: List of [x, y] points forming the lane contour
        class_id: Lane class ID from segmentation (1-4 for multi-class)
        confidence: Detection confidence [0, 1]
    """
    points: list  # List of [x, y] points
    class_id: int = 1
    confidence: float = 1.0

    @property
    def num_points(self) -> int:
        """Number of points in the contour."""
        return len(self.points)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'points': self.points,
            'class_id': self.class_id,
            'confidence': self.confidence
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LaneContour':
        """Create from dictionary."""
        return cls(
            points=data['points'],
            class_id=data.get('class_id', 1),
            confidence=data.get('confidence', 1.0)
        )



@dataclass
class LaneMetrics:
    """
    Lane analysis metrics.

    All measurements and derived values for lane keeping assist.
    """
    # Raw measurements
    vehicle_center_x: float | None = None
    lane_center_x: float | None = None
    lane_width_pixels: float | None = None

    # Calculated metrics
    lateral_offset_pixels: float | None = None
    lateral_offset_meters: float | None = None
    lateral_offset_normalized: float | None = None
    heading_angle_deg: float | None = None

    # Status
    departure_status: LaneDepartureStatus = LaneDepartureStatus.UNKNOWN

    # Metadata
    has_left_lane: bool = False
    has_right_lane: bool = False
    has_both_lanes: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility."""
        return {
            'vehicle_center_x': self.vehicle_center_x,
            'lane_center_x': self.lane_center_x,
            'lane_width_pixels': self.lane_width_pixels,
            'lateral_offset_pixels': self.lateral_offset_pixels,
            'lateral_offset_meters': self.lateral_offset_meters,
            'lateral_offset_normalized': self.lateral_offset_normalized,
            'heading_angle_deg': self.heading_angle_deg,
            'departure_status': self.departure_status,
            'has_left_lane': self.has_left_lane,
            'has_right_lane': self.has_right_lane,
            'has_both_lanes': self.has_both_lanes,
        }


@dataclass
class VehicleTelemetry:
    """Vehicle telemetry data."""
    speed_kmh: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    steering: float = 0.0
    gear: int = 0

    # Position (optional)
    location_x: float | None = None
    location_y: float | None = None
    location_z: float | None = None

    # Rotation (optional)
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None


@dataclass
class DetectionResult:
    """
    Complete result from lane detection process.

    Combines detected lanes with debug visualization.
    """
    left_lane: Lane | None = None
    right_lane: Lane | None = None
    debug_image: np.ndarray | None = None
    processing_time_ms: float = 0.0
    lanes: list | None = None  # List of LaneContour for DL multi-lane detection

    @property
    def has_left_lane(self) -> bool:
        return self.left_lane is not None

    @property
    def has_right_lane(self) -> bool:
        return self.right_lane is not None

    @property
    def has_both_lanes(self) -> bool:
        return self.left_lane is not None and self.right_lane is not None

    @property
    def num_lanes(self) -> int:
        """Get number of detected lane contours."""
        if self.lanes:
            return len(self.lanes)
        count = 0
        if self.left_lane:
            count += 1
        if self.right_lane:
            count += 1
        return count
