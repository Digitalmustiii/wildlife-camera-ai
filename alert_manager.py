"""
Wildlife Alert Management System
Smart notifications with cooldown, deduplication, and multiple channels
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class AlertEvent:
    """Represents an alert event"""
    species: str
    confidence: float
    rarity: str
    timestamp: datetime
    snapshot_path: Optional[str] = None
    bbox: tuple = field(default_factory=tuple)
    
    def to_dict(self) -> Dict:
        return {
            'species': self.species,
            'confidence': self.confidence,
            'rarity': self.rarity,
            'timestamp': self.timestamp.isoformat(),
            'snapshot_path': self.snapshot_path,
            'bbox': self.bbox
        }


class AlertManager:
    """Manages alert logic and notification dispatch"""
    
    def __init__(self, config):
        """
        Initialize alert manager
        
        Args:
            config: AlertConfig object with settings
        """
        self.config = config
        self.last_alert_time: Dict[str, datetime] = {}
        self.alert_count: Dict[str, int] = {}
        self.session_alerts: List[AlertEvent] = []
        self.handlers: List[Callable] = []
    
    def should_alert(self, detection: Dict) -> bool:
        """
        Determine if detection should trigger an alert
        
        Args:
            detection: Detection dict from detector
            
        Returns:
            True if alert should be sent
        """
        if not self.config.enabled:
            return False
        
        species = detection['species']
        rarity = detection['rarity']
        
        # Check rarity-based rules
        if rarity == 'common' and not self.config.alert_on_common:
            return False
        if rarity == 'interesting' and not self.config.alert_on_interesting:
            return False
        if rarity == 'rare' and not self.config.alert_on_rare:
            return False
        
        # Check cooldown
        if self._is_in_cooldown(species):
            return False
        
        return True
    
    def _is_in_cooldown(self, species: str) -> bool:
        """Check if species is in cooldown period"""
        if species not in self.last_alert_time:
            return False
        
        elapsed = (datetime.now() - self.last_alert_time[species]).total_seconds()
        return elapsed < self.config.cooldown_seconds
    
    def process_detection(self, detection: Dict, snapshot_path: Optional[str] = None) -> bool:
        """
        Process detection and send alert if needed
        
        Args:
            detection: Detection dict
            snapshot_path: Path to saved snapshot
            
        Returns:
            True if alert was sent
        """
        if not self.should_alert(detection):
            return False
        
        # Create alert event
        event = AlertEvent(
            species=detection['species'],
            confidence=detection['confidence'],
            rarity=detection['rarity'],
            timestamp=datetime.now(),
            snapshot_path=snapshot_path,
            bbox=detection['bbox']
        )
        
        # Send alert
        self._send_alert(event)
        
        # Update tracking
        species = detection['species']
        self.last_alert_time[species] = datetime.now()
        self.alert_count[species] = self.alert_count.get(species, 0) + 1
        self.session_alerts.append(event)
        
        return True
    
    def _send_alert(self, event: AlertEvent):
        """Dispatch alert to all registered handlers"""
        for handler in self.handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"⚠️  Alert handler error: {e}")
    
    def register_handler(self, handler: Callable):
        """
        Register alert handler function
        
        Args:
            handler: Function that takes AlertEvent as parameter
        """
        self.handlers.append(handler)
    
    def get_cooldown_status(self) -> Dict[str, float]:
        """Get remaining cooldown time for each species (in seconds)"""
        status = {}
        now = datetime.now()
        
        for species, last_time in self.last_alert_time.items():
            elapsed = (now - last_time).total_seconds()
            remaining = max(0, self.config.cooldown_seconds - elapsed)
            if remaining > 0:
                status[species] = remaining
        
        return status
    
    def get_session_summary(self) -> Dict:
        """Get summary of alerts sent this session"""
        species_breakdown = {}
        for alert in self.session_alerts:
            species_breakdown[alert.species] = species_breakdown.get(alert.species, 0) + 1
        
        return {
            'total_alerts': len(self.session_alerts),
            'unique_species': len(species_breakdown),
            'species_breakdown': species_breakdown,
            'rare_alerts': len([a for a in self.session_alerts if a.rarity == 'rare'])
        }
    
    def reset_session(self):
        """Reset session-specific alert tracking"""
        self.session_alerts.clear()
        self.alert_count.clear()
    
    def export_alerts(self, filepath: str):
        """Export session alerts to JSON"""
        data = [alert.to_dict() for alert in self.session_alerts]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Built-in alert handlers

def console_alert_handler(event: AlertEvent):
    """Print alert to console"""
    rarity_emoji = {'common': '🟢', 'interesting': '🟡', 'rare': '🔴'}
    emoji = rarity_emoji.get(event.rarity, '⚪')
    
    print(f"\n{'='*50}")
    print(f"🚨 ALERT: {event.species.upper()}")
    print(f"{'='*50}")
    print(f"{emoji} Rarity: {event.rarity}")
    print(f"📊 Confidence: {event.confidence:.2%}")
    print(f"🕐 Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    if event.snapshot_path:
        print(f"📸 Snapshot: {event.snapshot_path}")
    print(f"{'='*50}\n")


def file_alert_handler(log_file: str = "data/alerts.log"):
    """Create file logging handler"""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def handler(event: AlertEvent):
        with open(log_file, 'a') as f:
            f.write(f"[{event.timestamp.isoformat()}] "
                   f"{event.rarity.upper()} - {event.species} "
                   f"(conf: {event.confidence:.2f})\n")
    
    return handler


def webhook_alert_handler(webhook_url: str):
    """Create webhook handler (generic HTTP POST)"""
    import requests
    
    def handler(event: AlertEvent):
        try:
            payload = event.to_dict()
            response = requests.post(webhook_url, json=payload, timeout=5)
            if response.status_code != 200:
                print(f"⚠️  Webhook failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️  Webhook error: {e}")
    
    return handler


class AlertThrottle:
    """Advanced throttling for high-frequency detections"""
    
    def __init__(self, max_alerts_per_minute: int = 5):
        self.max_alerts = max_alerts_per_minute
        self.alert_times: List[datetime] = []
    
    def can_alert(self) -> bool:
        """Check if alert rate limit allows new alert"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        # Remove old alerts
        self.alert_times = [t for t in self.alert_times if t > cutoff]
        
        # Check limit
        if len(self.alert_times) >= self.max_alerts:
            return False
        
        self.alert_times.append(now)
        return True
    
    def get_remaining_quota(self) -> int:
        """Get number of alerts remaining in current minute"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        recent = len([t for t in self.alert_times if t > cutoff])
        return max(0, self.max_alerts - recent)