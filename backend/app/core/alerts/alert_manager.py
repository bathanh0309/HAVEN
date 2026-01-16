"""
Alert Manager - Handles alert logic, cooldown, and notification dispatch
"""
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from app.core.adl.actions import ADLAction, ADLLabel, SeverityLevel
from .telegram_bot import TelegramNotifier
from .templates import AlertTemplate


@dataclass
class AlertConfig:
    """Alert configuration"""
    # Cooldown periods (seconds) per severity level
    cooldown_low: int = 300  # 5 minutes
    cooldown_medium: int = 180  # 3 minutes
    cooldown_high: int = 60  # 1 minute
    cooldown_critical: int = 30  # 30 seconds (allow rapid alerts for falls)
    
    # Enable/disable notifications
    enable_telegram: bool = True
    enable_email: bool = False  # Future
    
    # Alert thresholds
    min_confidence_critical: float = 0.7  # Lower threshold for critical events
    min_confidence_normal: float = 0.8
    
    # Specific alert enables
    alert_on_fall: bool = True
    alert_on_stroke: bool = True
    alert_on_immobility: bool = True
    immobility_duration_threshold: int = 300  # 5 minutes


class AlertManager:
    """
    Manages alert lifecycle:
    1. Receives ADL events
    2. Applies business logic (cooldown, thresholds)
    3. Dispatches notifications
    4. Logs alerts to database
    """
    
    def __init__(self, config: AlertConfig = None, db_session=None):
        self.config = config or AlertConfig()
        self.db = db_session
        
        # Notification services
        self.telegram = TelegramNotifier() if self.config.enable_telegram else None
        
        # Cooldown tracking: {(camera_id, label): last_alert_time}
        self.last_alert_times: Dict[tuple, float] = defaultdict(float)
        
        # Alert deduplication: prevent spam for same event
        self.recent_alerts = {}  # {event_signature: timestamp}
        
    def process_event(self, action: ADLAction, camera_id: int, camera_location: str) -> Optional[Dict]:
        """
        Process ADL event and decide whether to send alert
        
        Args:
            action: Detected ADLAction
            camera_id: Source camera ID
            camera_location: Human-readable location (e.g., "Living Room")
        
        Returns:
            Alert info dict if alert sent, None otherwise
        """
        # Check if this event type should trigger alert
        if not self._should_alert(action):
            return None
        
        # Check confidence threshold
        min_conf = (self.config.min_confidence_critical 
                   if action.severity == SeverityLevel.CRITICAL 
                   else self.config.min_confidence_normal)
        
        if action.confidence < min_conf:
            return None
        
        # Check cooldown
        if not self._check_cooldown(camera_id, action.label, action.severity):
            return None
        
        # Check deduplication (same event signature within 5 seconds)
        event_sig = self._get_event_signature(action, camera_id)
        current_time = time.time()
        if event_sig in self.recent_alerts:
            if current_time - self.recent_alerts[event_sig] < 5:
                return None  # Duplicate event
        
        self.recent_alerts[event_sig] = current_time
        
        # Generate alert message
        alert_message = AlertTemplate.format_alert(
            action=action,
            camera_location=camera_location,
            timestamp=current_time
        )
        
        # Send notifications
        notifications_sent = []
        
        if self.telegram:
            try:
                self.telegram.send_alert(
                    message=alert_message["text"],
                    severity=action.severity.value,
                    snapshot_path=None  # TODO: attach snapshot from action or param
                )
                notifications_sent.append("telegram")
            except Exception as e:
                print(f"Failed to send Telegram alert: {e}")
        
        # Log alert to database
        alert_log = self._log_alert_to_db(
            action=action,
            camera_id=camera_id,
            camera_location=camera_location,
            message=alert_message["text"],
            notifications=notifications_sent
        )
        
        # Update cooldown
        self._update_cooldown(camera_id, action.label)
        
        return {
            "alert_id": alert_log.id if alert_log else None,
            "message": alert_message["text"],
            "severity": action.severity.value,
            "notifications_sent": notifications_sent,
            "timestamp": current_time
        }
    
    def _should_alert(self, action: ADLAction) -> bool:
        """Check if event type should trigger alert"""
        if action.label == ADLLabel.FALL and self.config.alert_on_fall:
            return True
        if action.label == ADLLabel.STROKE_LIKE and self.config.alert_on_stroke:
            return True
        # Add more conditions as needed, e.g. for general monitoring
        return False
    
    def _check_cooldown(self, camera_id: int, label: ADLLabel, severity: SeverityLevel) -> bool:
        """Check if cooldown period has passed"""
        key = (camera_id, label)
        last_time = self.last_alert_times.get(key, 0)
        current_time = time.time()
        
        # Determine cooldown based on severity
        cooldown_map = {
            SeverityLevel.LOW: self.config.cooldown_low,
            SeverityLevel.MEDIUM: self.config.cooldown_medium,
            SeverityLevel.HIGH: self.config.cooldown_high,
            SeverityLevel.CRITICAL: self.config.cooldown_critical
        }
        
        cooldown = cooldown_map.get(severity, 60)
        
        return (current_time - last_time) >= cooldown
    
    def _update_cooldown(self, camera_id: int, label: ADLLabel):
        """Update last alert time"""
        key = (camera_id, label)
        self.last_alert_times[key] = time.time()
    
    def _get_event_signature(self, action: ADLAction, camera_id: int) -> str:
        """Generate unique signature for event deduplication"""
        bbox_str = f"{action.bbox[0]},{action.bbox[1]},{action.bbox[2]},{action.bbox[3]}"
        return f"{camera_id}:{action.label.value}:{bbox_str}"
    
    def _log_alert_to_db(self, action, camera_id, camera_location, message, notifications):
        """
        Log alert to database
        """
        # TODO: Implement actual DB logging
        return None
