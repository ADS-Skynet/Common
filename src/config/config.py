"""
Configuration management for Skynet autonomous driving system.
Loads from YAML and provides type-safe access to settings.
"""

from dataclasses import dataclass, field
from typing import Tuple
import yaml
from pathlib import Path


def get_project_root() -> Path:
    """
    Find the project root directory by locating pyproject.toml.

    Returns:
        Path to project root directory
    """
    # Start from this file's location and walk up to find pyproject.toml
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback: assume project root is 3 levels up from this file
    # (config/config.py -> src -> sky_commmon -> project root)
    return Path(__file__).resolve().parent.parent.parent.parent


# Default config path at project root
DEFAULT_CONFIG_PATH = get_project_root() / "config.yaml"


@dataclass
class ZMQSocketConfig:
    """
    ZMQ socket-level configuration.

    Centralizes socket options that were hardcoded across multiple modules.
    """
    send_hwm: int = 10                    # SNDHWM - Send high water mark
    recv_hwm: int = 10                    # RCVHWM - Receive high water mark
    recv_timeout_ms: int = 100            # RCVTIMEO - Standard receive timeout
    param_recv_timeout_ms: int = 10       # RCVTIMEO for parameter updates (faster)
    linger_ms: int = 0                    # LINGER - Time to wait on close
    connection_delay_s: float = 0.1       # Delay after connection (slow-joiner fix)


@dataclass
class CommunicationConfig:
    """
    Inter-process communication configuration.

    Defines shared memory names and ZMQ ports for all modules.
    This ensures consistency across simulation, LKAS, and viewer.
    """
    # Shared Memory Channel Names
    image_shm_name: str = "camera_feed"
    detection_shm_name: str = "detection_results"
    control_shm_name: str = "control_commands"

    # ZMQ Base Ports
    zmq_broadcast_port: int = 5557      # Simulation -> Viewer (frame/detection/state)
    zmq_action_port: int = 5558         # Viewer -> Simulation (action commands)
    zmq_parameter_port: int = 5559      # Parameter updates

    # ZMQ Broker Derived Ports (used by LKAS broker)
    zmq_param_servers_port: int = 5560     # Parameter forwarding to servers
    zmq_action_forward_port: int = 5561    # Action forwarding to simulation
    zmq_vehicle_status_port: int = 5562    # Vehicle status broadcasting

    # Default hosts
    zmq_broadcast_host: str = "localhost"

    # Socket configuration
    zmq_socket: ZMQSocketConfig = field(default_factory=ZMQSocketConfig)


@dataclass
class CARLAConfig:
    """CARLA simulator configuration."""
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    vehicle_type: str = "vehicle.tesla.model3"


@dataclass
class RetryConfig:
    """
    Connection retry configuration.

    Centralizes retry parameters that were inconsistent across modules.
    """
    max_retries: int = 20                 # Standard retry count
    retry_delay_s: float = 0.5            # Standard retry delay
    extended_max_retries: int = 30        # For shared memory initialization
    extended_retry_delay_s: float = 1.0   # Extended retry delay


@dataclass
class TimingConfig:
    """
    System timing configuration.

    Centralizes timing parameters used across modules.
    """
    main_loop_sleep_s: float = 0.01       # Main loop sleep interval
    post_decision_delay_s: float = 0.5    # Delay after decision making
    pause_sleep_s: float = 0.1            # Sleep during pause state
    busy_wait_sleep_s: float = 0.001      # Shared memory busy-wait interval
    warmup_duration_s: float = 2.5        # System warmup duration


@dataclass
class StreamingConfig:
    """
    Video streaming and compression configuration.

    Centralizes streaming parameters used across modules.
    """
    jpeg_quality: int = 85                # JPEG compression quality (0-100)
    raw_rgb: bool = False                 # Send raw RGB instead of JPEG (faster for localhost)
    web_viewer_fps: int = 30              # Web viewer target FPS
    stream_frame_delay_ms: int = 33       # Frame delay for streaming (~30fps)
    broadcast_log_interval: int = 100     # Log every N frames


@dataclass
class LauncherConfig:
    """
    Process launcher configuration.

    Centralizes process management parameters.
    """
    decision_init_timeout_s: float = 3.0   # Decision process init timeout
    detection_init_timeout_s: float = 4.0  # Detection process init timeout
    process_stop_timeout_s: float = 5.0    # Process termination timeout
    terminal_width: int = 70               # Terminal output width
    subprocess_prefix: str = "[SubProc]"   # Subprocess log prefix
    log_file: str = "lkas_run.log"         # Default log file name
    buffer_read_size: int = 4096           # Subprocess output buffer size


@dataclass
class ControlLimitsConfig:
    """
    Control signal limits.

    Defines bounds for steering, throttle, and brake signals.
    """
    max_steering: float = 1.0              # Maximum steering angle (normalized)
    min_steering: float = -1.0             # Minimum steering angle (normalized)
    max_throttle: float = 1.0              # Maximum throttle
    min_throttle: float = 0.0              # Minimum throttle
    max_brake: float = 1.0                 # Maximum brake
    min_brake: float = 0.0                 # Minimum brake


@dataclass
class CameraConfig:
    """Camera sensor configuration."""
    width: int = 640
    height: int = 480
    fov: float = 90.0
    position: Tuple[float, float, float] = (2.0, 0.0, 1.5)  # x, y, z
    rotation: Tuple[float, float, float] = (-10.0, 0.0, 0.0)  # pitch, yaw, roll


@dataclass
class CVDetectorConfig:
    """Computer Vision detector parameters."""
    canny_low: int = 50
    canny_high: int = 150
    hough_rho: int = 2
    hough_theta: float = 0.017453  # pi/180
    hough_threshold: int = 50
    hough_min_line_len: int = 40
    hough_max_line_gap: int = 100
    smoothing_factor: float = 0.7
    min_slope: float = 0.5

    # ROI configuration (broader detection area)
    roi_bottom_left_x: float = 0.05  # fraction of width (bottom-left corner)
    roi_top_left_x: float = 0.35     # fraction of width (top-left corner) - wider than before
    roi_top_right_x: float = 0.65    # fraction of width (top-right corner) - wider than before
    roi_bottom_right_x: float = 0.95 # fraction of width (bottom-right corner)
    roi_top_y: float = 0.5           # fraction of height (look at top 50% of image)


@dataclass
class DLDetectorConfig:
    """Deep Learning detector parameters."""
    model_type: str = "pretrained"  # 'pretrained', 'simple', 'full'
    input_size: Tuple[int, int] = (256, 256)
    threshold: float = 0.5
    device: str = "auto"  # 'cpu', 'cuda', 'auto'


@dataclass
class AnalyzerConfig:
    """Lane analyzer configuration."""
    drift_threshold: float = 0.15
    departure_threshold: float = 0.35
    lane_width_meters: float = 3.7
    max_heading_degrees: float = 30.0


@dataclass
class ControllerConfig:
    """PD controller parameters."""
    kp: float = 0.5  # Proportional gain
    kd: float = 0.1  # Derivative gain


@dataclass
class ThrottlePolicyConfig:
    """Adaptive throttle policy configuration."""
    base: float = 0.14              # Base throttle when steering is minimal
    min: float = 0.05               # Minimum throttle during sharp turns
    steer_threshold: float = 0.15   # Steering magnitude to start reducing throttle
    steer_max: float = 0.70         # Maximum expected steering magnitude


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    show_spectator_overlay: bool = True
    follow_with_spectator: bool = False
    alert_blink_frequency: int = 10

    # Web viewer port
    web_port: int = 8080

    # Colors (BGR format for OpenCV)
    color_left_lane: Tuple[int, int, int] = (255, 0, 0)  # Blue
    color_right_lane: Tuple[int, int, int] = (0, 0, 255)  # Red
    color_lane_fill: Tuple[int, int, int] = (0, 255, 0)  # Green
    color_centered: Tuple[int, int, int] = (0, 255, 0)  # Green
    color_drift: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    color_departure: Tuple[int, int, int] = (0, 0, 255)  # Red

    # HUD settings
    hud_font_scale: float = 0.6
    hud_thickness: int = 2
    hud_margin: int = 20


@dataclass
class Config:
    """
    Master configuration container.

    Aggregates all subsystem configurations.
    """
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    carla: CARLAConfig = field(default_factory=CARLAConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    cv_detector: CVDetectorConfig = field(default_factory=CVDetectorConfig)
    dl_detector: DLDetectorConfig = field(default_factory=DLDetectorConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    throttle_policy: ThrottlePolicyConfig = field(default_factory=ThrottlePolicyConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)
    control_limits: ControlLimitsConfig = field(default_factory=ControlLimitsConfig)

    # General settings
    detection_method: str = "cv"  # 'cv' or 'dl'


class ConfigManager:
    """
    Configuration manager with YAML loading.

    Usage:
        # Load from project root config.yaml (default)
        config = ConfigManager.load()

        # Load from specific path
        config = ConfigManager.load('path/to/config.yaml')

        # Use built-in defaults only
        config = ConfigManager.load('default')

        host = config.carla.host
    """

    @staticmethod
    def load(config_path: str | Path | None = None) -> Config:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file.
                        If None, tries to load from project root config.yaml.
                        If "default", uses built-in defaults without loading file.

        Returns:
            Config object with loaded settings
        """
        # If explicitly asked for defaults, return without loading
        if config_path == "default":
            return Config()

        # If no path given, use project root config.yaml
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        path = Path(config_path)
        if not path.exists():
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return Config()

        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)

            if data is None:
                return Config()

            # Parse communication config
            comm_cfg = CommunicationConfig()
            if 'communication' in data:
                comm_data = data['communication']

                # Parse ZMQ socket config if present
                zmq_socket_cfg = ZMQSocketConfig()
                if 'zmq_socket' in comm_data:
                    sock_data = comm_data['zmq_socket']
                    zmq_socket_cfg = ZMQSocketConfig(
                        send_hwm=sock_data.get('send_hwm', zmq_socket_cfg.send_hwm),
                        recv_hwm=sock_data.get('recv_hwm', zmq_socket_cfg.recv_hwm),
                        recv_timeout_ms=sock_data.get('recv_timeout_ms', zmq_socket_cfg.recv_timeout_ms),
                        param_recv_timeout_ms=sock_data.get('param_recv_timeout_ms', zmq_socket_cfg.param_recv_timeout_ms),
                        linger_ms=sock_data.get('linger_ms', zmq_socket_cfg.linger_ms),
                        connection_delay_s=sock_data.get('connection_delay_s', zmq_socket_cfg.connection_delay_s),
                    )

                comm_cfg = CommunicationConfig(
                    image_shm_name=comm_data.get('image_shm_name', comm_cfg.image_shm_name),
                    detection_shm_name=comm_data.get('detection_shm_name', comm_cfg.detection_shm_name),
                    control_shm_name=comm_data.get('control_shm_name', comm_cfg.control_shm_name),
                    zmq_broadcast_port=comm_data.get('zmq_broadcast_port', comm_cfg.zmq_broadcast_port),
                    zmq_action_port=comm_data.get('zmq_action_port', comm_cfg.zmq_action_port),
                    zmq_parameter_port=comm_data.get('zmq_parameter_port', comm_cfg.zmq_parameter_port),
                    zmq_param_servers_port=comm_data.get('zmq_param_servers_port', comm_cfg.zmq_param_servers_port),
                    zmq_action_forward_port=comm_data.get('zmq_action_forward_port', comm_cfg.zmq_action_forward_port),
                    zmq_vehicle_status_port=comm_data.get('zmq_vehicle_status_port', comm_cfg.zmq_vehicle_status_port),
                    zmq_broadcast_host=comm_data.get('zmq_broadcast_host', comm_cfg.zmq_broadcast_host),
                    zmq_socket=zmq_socket_cfg,
                )

            # Parse CARLA config
            carla_cfg = CARLAConfig()
            if 'carla' in data:
                carla_data = data['carla']
                carla_cfg = CARLAConfig(
                    host=carla_data.get('host', carla_cfg.host),
                    port=carla_data.get('port', carla_cfg.port),
                    timeout=carla_data.get('timeout', carla_cfg.timeout),
                    vehicle_type=carla_data.get('vehicle_type', carla_cfg.vehicle_type),
                )

            # Parse camera config
            camera_cfg = CameraConfig()
            if 'camera' in data:
                cam_data = data['camera']

                # Convert position dict to tuple if needed
                position = cam_data.get('position', camera_cfg.position)
                if isinstance(position, dict):
                    position = (position['x'], position['y'], position['z'])
                elif not isinstance(position, tuple):
                    position = tuple(position)

                # Convert rotation dict to tuple if needed
                rotation = cam_data.get('rotation', camera_cfg.rotation)
                if isinstance(rotation, dict):
                    rotation = (rotation['pitch'], rotation['yaw'], rotation['roll'])
                elif not isinstance(rotation, tuple):
                    rotation = tuple(rotation)

                camera_cfg = CameraConfig(
                    width=cam_data.get('width', camera_cfg.width),
                    height=cam_data.get('height', camera_cfg.height),
                    fov=cam_data.get('fov', camera_cfg.fov),
                    position=position,
                    rotation=rotation,
                )

            # Parse CV detector config
            cv_cfg = CVDetectorConfig()
            if 'cv_detector' in data:
                cv_data = data['cv_detector']
                cv_cfg = CVDetectorConfig(
                    canny_low=cv_data.get('canny_low', cv_cfg.canny_low),
                    canny_high=cv_data.get('canny_high', cv_cfg.canny_high),
                    hough_rho=cv_data.get('hough_rho', cv_cfg.hough_rho),
                    hough_theta=cv_data.get('hough_theta', cv_cfg.hough_theta),
                    hough_threshold=cv_data.get('hough_threshold', cv_cfg.hough_threshold),
                    hough_min_line_len=cv_data.get('hough_min_line_len', cv_cfg.hough_min_line_len),
                    hough_max_line_gap=cv_data.get('hough_max_line_gap', cv_cfg.hough_max_line_gap),
                    smoothing_factor=cv_data.get('smoothing_factor', cv_cfg.smoothing_factor),
                    min_slope=cv_data.get('min_slope', cv_cfg.min_slope),
                    roi_bottom_left_x=cv_data.get('roi_bottom_left_x', cv_cfg.roi_bottom_left_x),
                    roi_top_left_x=cv_data.get('roi_top_left_x', cv_cfg.roi_top_left_x),
                    roi_top_right_x=cv_data.get('roi_top_right_x', cv_cfg.roi_top_right_x),
                    roi_bottom_right_x=cv_data.get('roi_bottom_right_x', cv_cfg.roi_bottom_right_x),
                    roi_top_y=cv_data.get('roi_top_y', cv_cfg.roi_top_y),
                )

            # Parse DL detector config
            dl_cfg = DLDetectorConfig()
            if 'dl_detector' in data:
                dl_data = data['dl_detector']

                # Convert input_size list to tuple if needed
                input_size = dl_data.get('input_size', dl_cfg.input_size)
                if isinstance(input_size, list):
                    input_size = tuple(input_size)

                dl_cfg = DLDetectorConfig(
                    model_type=dl_data.get('model_type', dl_cfg.model_type),
                    input_size=input_size,
                    threshold=dl_data.get('threshold', dl_cfg.threshold),
                    device=dl_data.get('device', dl_cfg.device),
                )

            # Parse analyzer config
            analyzer_cfg = AnalyzerConfig()
            if 'lane_analyzer' in data:
                ana_data = data['lane_analyzer']
                analyzer_cfg = AnalyzerConfig(
                    drift_threshold=ana_data.get('drift_threshold', analyzer_cfg.drift_threshold),
                    departure_threshold=ana_data.get('departure_threshold', analyzer_cfg.departure_threshold),
                    lane_width_meters=ana_data.get('lane_width_meters', analyzer_cfg.lane_width_meters),
                    max_heading_degrees=ana_data.get('max_heading_degrees', analyzer_cfg.max_heading_degrees),
                )

            # Parse controller config
            controller_cfg = ControllerConfig()
            if 'lane_analyzer' in data:
                ctrl_data = data['lane_analyzer']
                controller_cfg = ControllerConfig(
                    kp=ctrl_data.get('kp', controller_cfg.kp),
                    kd=ctrl_data.get('kd', controller_cfg.kd),
                )

            # Parse throttle policy config
            throttle_cfg = ThrottlePolicyConfig()
            if 'throttle_policy' in data:
                throttle_data = data['throttle_policy']
                throttle_cfg = ThrottlePolicyConfig(
                    base=throttle_data.get('base', throttle_cfg.base),
                    min=throttle_data.get('min', throttle_cfg.min),
                    steer_threshold=throttle_data.get('steer_threshold', throttle_cfg.steer_threshold),
                    steer_max=throttle_data.get('steer_max', throttle_cfg.steer_max),
                )

            # Parse visualization config
            viz_cfg = VisualizationConfig()
            if 'visualization' in data:
                viz_data = data['visualization']

                # Parse colors if present (convert list to tuple)
                def parse_color(color_val, default):
                    if isinstance(color_val, list):
                        return tuple(color_val)
                    return default

                viz_cfg = VisualizationConfig(
                    show_spectator_overlay=viz_data.get('show_spectator_overlay', viz_cfg.show_spectator_overlay),
                    follow_with_spectator=viz_data.get('follow_with_spectator', viz_cfg.follow_with_spectator),
                    alert_blink_frequency=viz_data.get('alert_blink_frequency', viz_cfg.alert_blink_frequency),
                    color_left_lane=parse_color(viz_data.get('color_left_lane'), viz_cfg.color_left_lane),
                    color_right_lane=parse_color(viz_data.get('color_right_lane'), viz_cfg.color_right_lane),
                    color_lane_fill=parse_color(viz_data.get('color_lane_fill'), viz_cfg.color_lane_fill),
                    color_centered=parse_color(viz_data.get('color_centered'), viz_cfg.color_centered),
                    color_drift=parse_color(viz_data.get('color_drift'), viz_cfg.color_drift),
                    color_departure=parse_color(viz_data.get('color_departure'), viz_cfg.color_departure),
                    hud_font_scale=viz_data.get('hud_font_scale', viz_cfg.hud_font_scale),
                    hud_thickness=viz_data.get('hud_thickness', viz_cfg.hud_thickness),
                    hud_margin=viz_data.get('hud_margin', viz_cfg.hud_margin),
                )

            # Parse retry config
            retry_cfg = RetryConfig()
            if 'retry' in data:
                retry_data = data['retry']
                retry_cfg = RetryConfig(
                    max_retries=retry_data.get('max_retries', retry_cfg.max_retries),
                    retry_delay_s=retry_data.get('retry_delay_s', retry_cfg.retry_delay_s),
                    extended_max_retries=retry_data.get('extended_max_retries', retry_cfg.extended_max_retries),
                    extended_retry_delay_s=retry_data.get('extended_retry_delay_s', retry_cfg.extended_retry_delay_s),
                )

            # Parse timing config
            timing_cfg = TimingConfig()
            if 'timing' in data:
                timing_data = data['timing']
                timing_cfg = TimingConfig(
                    main_loop_sleep_s=timing_data.get('main_loop_sleep_s', timing_cfg.main_loop_sleep_s),
                    post_decision_delay_s=timing_data.get('post_decision_delay_s', timing_cfg.post_decision_delay_s),
                    pause_sleep_s=timing_data.get('pause_sleep_s', timing_cfg.pause_sleep_s),
                    busy_wait_sleep_s=timing_data.get('busy_wait_sleep_s', timing_cfg.busy_wait_sleep_s),
                    warmup_duration_s=timing_data.get('warmup_duration_s', timing_cfg.warmup_duration_s),
                )

            # Parse streaming config
            streaming_cfg = StreamingConfig()
            if 'streaming' in data:
                stream_data = data['streaming']
                streaming_cfg = StreamingConfig(
                    jpeg_quality=stream_data.get('jpeg_quality', streaming_cfg.jpeg_quality),
                    raw_rgb=stream_data.get('raw_rgb', streaming_cfg.raw_rgb),
                    web_viewer_fps=stream_data.get('web_viewer_fps', streaming_cfg.web_viewer_fps),
                    stream_frame_delay_ms=stream_data.get('stream_frame_delay_ms', streaming_cfg.stream_frame_delay_ms),
                    broadcast_log_interval=stream_data.get('broadcast_log_interval', streaming_cfg.broadcast_log_interval),
                )

            # Parse launcher config
            launcher_cfg = LauncherConfig()
            if 'launcher' in data:
                launcher_data = data['launcher']
                launcher_cfg = LauncherConfig(
                    decision_init_timeout_s=launcher_data.get('decision_init_timeout_s', launcher_cfg.decision_init_timeout_s),
                    detection_init_timeout_s=launcher_data.get('detection_init_timeout_s', launcher_cfg.detection_init_timeout_s),
                    process_stop_timeout_s=launcher_data.get('process_stop_timeout_s', launcher_cfg.process_stop_timeout_s),
                    terminal_width=launcher_data.get('terminal_width', launcher_cfg.terminal_width),
                    subprocess_prefix=launcher_data.get('subprocess_prefix', launcher_cfg.subprocess_prefix),
                    log_file=launcher_data.get('log_file', launcher_cfg.log_file),
                    buffer_read_size=launcher_data.get('buffer_read_size', launcher_cfg.buffer_read_size),
                )

            # Parse control limits config
            control_limits_cfg = ControlLimitsConfig()
            if 'control_limits' in data:
                limits_data = data['control_limits']
                control_limits_cfg = ControlLimitsConfig(
                    max_steering=limits_data.get('max_steering', control_limits_cfg.max_steering),
                    min_steering=limits_data.get('min_steering', control_limits_cfg.min_steering),
                    max_throttle=limits_data.get('max_throttle', control_limits_cfg.max_throttle),
                    min_throttle=limits_data.get('min_throttle', control_limits_cfg.min_throttle),
                    max_brake=limits_data.get('max_brake', control_limits_cfg.max_brake),
                    min_brake=limits_data.get('min_brake', control_limits_cfg.min_brake),
                )

            # Parse detection method from system section
            detection_method = "cv"
            if 'system' in data:
                detection_method = data['system'].get('detection_method', 'cv')

            # Create config object
            config = Config(
                communication=comm_cfg,
                carla=carla_cfg,
                camera=camera_cfg,
                cv_detector=cv_cfg,
                dl_detector=dl_cfg,
                analyzer=analyzer_cfg,
                controller=controller_cfg,
                throttle_policy=throttle_cfg,
                visualization=viz_cfg,
                retry=retry_cfg,
                timing=timing_cfg,
                streaming=streaming_cfg,
                launcher=launcher_cfg,
                control_limits=control_limits_cfg,
                detection_method=detection_method,
            )

            return config

        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration.")
            return Config()

    @staticmethod
    def save(config: Config, config_path: str) -> bool:
        """
        Save configuration to YAML file.

        Args:
            config: Config object to save
            config_path: Path to save YAML file

        Returns:
            True if successful
        """
        try:
            # Convert position and rotation tuples to dict format
            position = config.camera.position
            rotation = config.camera.rotation

            data = {
                'communication': {
                    'image_shm_name': config.communication.image_shm_name,
                    'detection_shm_name': config.communication.detection_shm_name,
                    'control_shm_name': config.communication.control_shm_name,
                    'zmq_broadcast_port': config.communication.zmq_broadcast_port,
                    'zmq_action_port': config.communication.zmq_action_port,
                    'zmq_parameter_port': config.communication.zmq_parameter_port,
                    'zmq_param_servers_port': config.communication.zmq_param_servers_port,
                    'zmq_action_forward_port': config.communication.zmq_action_forward_port,
                    'zmq_vehicle_status_port': config.communication.zmq_vehicle_status_port,
                    'zmq_broadcast_host': config.communication.zmq_broadcast_host,
                    'zmq_socket': {
                        'send_hwm': config.communication.zmq_socket.send_hwm,
                        'recv_hwm': config.communication.zmq_socket.recv_hwm,
                        'recv_timeout_ms': config.communication.zmq_socket.recv_timeout_ms,
                        'param_recv_timeout_ms': config.communication.zmq_socket.param_recv_timeout_ms,
                        'linger_ms': config.communication.zmq_socket.linger_ms,
                        'connection_delay_s': config.communication.zmq_socket.connection_delay_s,
                    },
                },
                'carla': {
                    'host': config.carla.host,
                    'port': config.carla.port,
                    'timeout': config.carla.timeout,
                    'vehicle_type': config.carla.vehicle_type,
                },
                'camera': {
                    'width': config.camera.width,
                    'height': config.camera.height,
                    'fov': config.camera.fov,
                    'position': {
                        'x': position[0],
                        'y': position[1],
                        'z': position[2],
                    },
                    'rotation': {
                        'pitch': rotation[0],
                        'yaw': rotation[1],
                        'roll': rotation[2],
                    },
                },
                'cv_detector': {
                    'canny_low': config.cv_detector.canny_low,
                    'canny_high': config.cv_detector.canny_high,
                    'hough_rho': config.cv_detector.hough_rho,
                    'hough_theta': config.cv_detector.hough_theta,
                    'hough_threshold': config.cv_detector.hough_threshold,
                    'hough_min_line_len': config.cv_detector.hough_min_line_len,
                    'hough_max_line_gap': config.cv_detector.hough_max_line_gap,
                    'smoothing_factor': config.cv_detector.smoothing_factor,
                    'min_slope': config.cv_detector.min_slope,
                    'roi_bottom_left_x': config.cv_detector.roi_bottom_left_x,
                    'roi_top_left_x': config.cv_detector.roi_top_left_x,
                    'roi_top_right_x': config.cv_detector.roi_top_right_x,
                    'roi_bottom_right_x': config.cv_detector.roi_bottom_right_x,
                    'roi_top_y': config.cv_detector.roi_top_y,
                },
                'dl_detector': {
                    'model_type': config.dl_detector.model_type,
                    'input_size': list(config.dl_detector.input_size),
                    'threshold': config.dl_detector.threshold,
                    'device': config.dl_detector.device,
                },
                'lane_analyzer': {
                    'drift_threshold': config.analyzer.drift_threshold,
                    'departure_threshold': config.analyzer.departure_threshold,
                    'lane_width_meters': config.analyzer.lane_width_meters,
                    'max_heading_degrees': config.analyzer.max_heading_degrees,
                    'kp': config.controller.kp,
                    'kd': config.controller.kd,
                },
                'throttle_policy': {
                    'base': config.throttle_policy.base,
                    'min': config.throttle_policy.min,
                    'steer_threshold': config.throttle_policy.steer_threshold,
                    'steer_max': config.throttle_policy.steer_max,
                },
                'visualization': {
                    'show_spectator_overlay': config.visualization.show_spectator_overlay,
                    'follow_with_spectator': config.visualization.follow_with_spectator,
                    'alert_blink_frequency': config.visualization.alert_blink_frequency,
                    'color_left_lane': list(config.visualization.color_left_lane),
                    'color_right_lane': list(config.visualization.color_right_lane),
                    'color_lane_fill': list(config.visualization.color_lane_fill),
                    'color_centered': list(config.visualization.color_centered),
                    'color_drift': list(config.visualization.color_drift),
                    'color_departure': list(config.visualization.color_departure),
                    'hud_font_scale': config.visualization.hud_font_scale,
                    'hud_thickness': config.visualization.hud_thickness,
                    'hud_margin': config.visualization.hud_margin,
                },
                'retry': {
                    'max_retries': config.retry.max_retries,
                    'retry_delay_s': config.retry.retry_delay_s,
                    'extended_max_retries': config.retry.extended_max_retries,
                    'extended_retry_delay_s': config.retry.extended_retry_delay_s,
                },
                'timing': {
                    'main_loop_sleep_s': config.timing.main_loop_sleep_s,
                    'post_decision_delay_s': config.timing.post_decision_delay_s,
                    'pause_sleep_s': config.timing.pause_sleep_s,
                    'busy_wait_sleep_s': config.timing.busy_wait_sleep_s,
                    'warmup_duration_s': config.timing.warmup_duration_s,
                },
                'streaming': {
                    'jpeg_quality': config.streaming.jpeg_quality,
                    'web_viewer_fps': config.streaming.web_viewer_fps,
                    'stream_frame_delay_ms': config.streaming.stream_frame_delay_ms,
                    'broadcast_log_interval': config.streaming.broadcast_log_interval,
                },
                'launcher': {
                    'decision_init_timeout_s': config.launcher.decision_init_timeout_s,
                    'detection_init_timeout_s': config.launcher.detection_init_timeout_s,
                    'process_stop_timeout_s': config.launcher.process_stop_timeout_s,
                    'terminal_width': config.launcher.terminal_width,
                    'subprocess_prefix': config.launcher.subprocess_prefix,
                    'log_file': config.launcher.log_file,
                    'buffer_read_size': config.launcher.buffer_read_size,
                },
                'control_limits': {
                    'max_steering': config.control_limits.max_steering,
                    'min_steering': config.control_limits.min_steering,
                    'max_throttle': config.control_limits.max_throttle,
                    'min_throttle': config.control_limits.min_throttle,
                    'max_brake': config.control_limits.max_brake,
                    'min_brake': config.control_limits.min_brake,
                },
                'system': {
                    'detection_method': config.detection_method,
                },
            }

            with open(config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)

            return True

        except Exception as e:
            print(f"Error saving config: {e}")
            return False
