"""
Alert message templates
"""
from typing import Dict, Any

class AlertTemplate:
    @staticmethod
    def format_alert(action, camera_location: str, timestamp: float) -> Dict[str, str]:
        """
        Format the alert message
        """
        from datetime import datetime
        dt_string = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S %d/%m/%Y")
        
        text = (
            f"<b>Activity Detected: {action.label.value.upper()}</b>\n"
            f"📍 Location: {camera_location}\n"
            f"🕒 Time: {dt_string}\n"
            f"🎯 Confidence: {action.confidence:.2f}\n"
            f"⚠️ Severity: {action.severity.value.upper()}"
        )
        
        return {"text": text}
