"""
Wildlife Camera Configuration
Centralized settings for detection, alerts, and recording
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class DetectionConfig:
    """Detection model settings"""
    model_size: str = 'n'  # n, s, m, l, x
    confidence_threshold: float = 0.5
    input_size: int = 640  # YOLO input resolution
    
    def validate(self):
        assert self.model_size in ['n', 's', 'm', 'l', 'x'], "Invalid model size"
        assert 0.0 <= self.confidence_threshold <= 1.0, "Confidence must be 0-1"


@dataclass
class VideoConfig:
    """Video input/output settings"""
    source: int | str = 0  # 0 for webcam, or video file path
    display_width: int = 1280
    display_height: int = 720
    record_fps: int = 30
    
    @property
    def is_webcam(self) -> bool:
        return isinstance(self.source, int)
    
    @property
    def is_file(self) -> bool:
        return isinstance(self.source, str) and Path(self.source).exists()


@dataclass
class AlertConfig:
    """Alert notification settings"""
    enabled: bool = True
    cooldown_seconds: int = 30  # Min time between alerts for same species
    
    # Alert triggers by rarity
    alert_on_common: bool = False
    alert_on_interesting: bool = True
    alert_on_rare: bool = True
    
    # Notification channels
    telegram_enabled: bool = False
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    email_enabled: bool = False
    email_to: Optional[str] = None


@dataclass
class RecordingConfig:
    """Video recording settings"""
    enabled: bool = True
    clip_duration_seconds: int = 15
    save_snapshots: bool = True
    
    # Storage paths
    clips_dir: Path = Path('data/clips')
    snapshots_dir: Path = Path('data/snapshots')
    database_path: Path = Path('data/wildlife.db')
    
    def create_directories(self):
        """Ensure storage directories exist"""
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class DisplayConfig:
    """UI display settings"""
    show_fps: bool = True
    show_confidence: bool = True
    show_stats_overlay: bool = True
    window_name: str = "Wildlife Camera AI"


class Config:
    """Master configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.detection = DetectionConfig()
        self.video = VideoConfig()
        self.alert = AlertConfig()
        self.recording = RecordingConfig()
        self.display = DisplayConfig()
        
        if config_file and Path(config_file).exists():
            self.load(config_file)
        
        # Validate and setup
        self.detection.validate()
        self.recording.create_directories()
    
    def load(self, filepath: str):
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for section, values in data.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        # Convert string paths to Path objects
                        if 'path' in key or 'dir' in key:
                            value = Path(value)
                        setattr(config_obj, key, value)
    
    def save(self, filepath: str):
        """Save config to JSON file"""
        data = {}
        for attr in ['detection', 'video', 'alert', 'recording', 'display']:
            config_obj = getattr(self, attr)
            data[attr] = {}
            for key, value in config_obj.__dict__.items():
                # Convert Path to string for JSON
                if isinstance(value, Path):
                    value = str(value)
                data[attr][key] = value
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def __repr__(self) -> str:
        """Pretty print config"""
        lines = ["Wildlife Camera Configuration:"]
        for attr in ['detection', 'video', 'alert', 'recording', 'display']:
            lines.append(f"\n[{attr.upper()}]")
            config_obj = getattr(self, attr)
            for key, value in config_obj.__dict__.items():
                lines.append(f"  {key}: {value}")
        return '\n'.join(lines)


# Preset configurations
class Presets:
    """Common configuration presets"""
    
    @staticmethod
    def laptop_demo() -> Config:
        """Optimized for laptop demo/interview"""
        config = Config()
        config.detection.model_size = 'n'
        config.detection.confidence_threshold = 0.5
        config.video.source = 0  # Webcam
        config.alert.enabled = True  # Alerts enabled
        config.alert.alert_on_common = True  # Alert on all animals
        config.recording.enabled = True  # Auto-save snapshots
        return config
    
    @staticmethod
    def video_file(filepath: str) -> Config:
        """Process pre-recorded video file"""
        config = Config()
        config.video.source = filepath
        config.alert.alert_on_common = True  # Log everything
        config.recording.enabled = True
        return config
    
    @staticmethod
    def production_pi() -> Config:
        """Optimized for Raspberry Pi deployment"""
        config = Config()
        config.detection.model_size = 'n'
        config.detection.confidence_threshold = 0.6  # Reduce false positives
        config.video.source = 0
        config.alert.enabled = True
        config.alert.telegram_enabled = True
        config.recording.enabled = True
        return config


# Quick access to presets
def get_config(preset: str = 'laptop_demo', **kwargs) -> Config:
    """
    Get configuration by preset name
    
    Args:
        preset: 'laptop_demo', 'video_file', 'production_pi', or 'custom'
        **kwargs: Override specific settings
        
    Returns:
        Config object
    """
    if preset == 'laptop_demo':
        config = Presets.laptop_demo()
    elif preset == 'video_file':
        filepath = kwargs.get('filepath', 'wildlife.mp4')
        config = Presets.video_file(filepath)
    elif preset == 'production_pi':
        config = Presets.production_pi()
    else:
        config = Config()
    
    # Apply overrides
    for key, value in kwargs.items():
        section, attr = key.split('_', 1) if '_' in key else ('detection', key)
        if hasattr(config, section):
            setattr(getattr(config, section), attr, value)
    
    return config