# Skynet Common

Shared types and utilities for the Skynet Autonomous Driving System.

## Overview

This package provides platform-independent shared components that can be used across all Skynet modules without requiring heavy dependencies like CARLA or PyTorch.

## Features

- **Data Models**: Lane detection types, vehicle telemetry, metrics
- **ZMQ Communication**: Pub-sub messaging for distributed systems
- **Visualization**: LKAS overlay and HUD rendering

## Installation

```bash
pip install -e .
```

Or from GitHub:
```bash
pip install git+https://github.com/ADS-Skynet/Common.git
```

## Usage

### Data Types
```python
from common.types import Lane, LaneDepartureStatus, LaneMetrics

lane = Lane(x1=100, y1=400, x2=200, y2=100, confidence=0.95)
status = LaneDepartureStatus.CENTERED
```

### ZMQ Communication
```python
from common.communication import ViewerSubscriber, DetectionData, VehicleState

subscriber = ViewerSubscriber("tcp://vehicle-ip:5557")
subscriber.register_frame_callback(on_frame)
subscriber.run_loop()
```

### Visualization
```python
from common.visualization import LKASVisualizer

visualizer = LKASVisualizer()
output = visualizer.draw_lanes(image, left_lane, right_lane)
output = visualizer.draw_hud(output, metrics, steering_value=0.1)
```

## Dependencies

This package has minimal dependencies:
- numpy
- opencv-python
- pyzmq
- rich

**No CARLA, PyTorch, or other heavy ML frameworks required!**

This makes it perfect for running on platforms like M1 Mac where CARLA is not supported.

## Package Structure

```
common/
├── types/           # Data models (Lane, LaneDepartureStatus, etc.)
├── communication/   # ZMQ pub-sub utilities
└── visualization/   # LKAS visualizer
```
