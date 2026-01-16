"""
Telegram Bot Notifier
"""
import requests
from typing import Optional
import os
from datetime import datetime

# from app.core.adl.actions import SeverityLevel # Circular import if we use type hint directly here without string

class TelegramNotifier:
    """
    Sends alerts via Telegram Bot API
    Adapted from user's provided code
    """
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN") or "8383210571:AAEfg3IIBtTVI_PcmfJ4w5uYgeM8thWqTPs"
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID") or "7827433045"
        
        if not self.token or not self.chat_id:
            print("Telegram warning: Token or Chat ID missing")
        
        self.api_url = f"https://api.telegram.org/bot{self.token}"

    def send_alert(self, message: str, severity: str, snapshot_path: Optional[str] = None):
        """
        Send alert message to Telegram
        
        Args:
            message: Alert text
            severity: Severity level (low, medium, high, critical)
            snapshot_path: Optional image path to attach
        """
        # Add severity emoji
        emoji_map = {
            "low": "ℹ️",
            "medium": "⚠️",
            "high": "🚨",
            "critical": "🆘"
        }
        
        emoji = emoji_map.get(str(severity).lower(), "📢")
        formatted_message = f"{emoji} <b>HAVEN Alert</b>\n\n{message}"
        
        if snapshot_path:
             self._send_photo(snapshot_path, formatted_message)
        else:
             self._send_message(formatted_message)
    
    def _send_message(self, message):
        """Internal: Send text message"""
        try:
            url = f"{self.api_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            print(f"[Telegram] Error sending message: {e}")

    def _send_photo(self, image_path, caption):
        """Internal: Send photo with caption"""
        if not os.path.exists(image_path):
             self._send_message(caption) # Fallback to text
             return

        try:
            url = f"{self.api_url}/sendPhoto"
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.chat_id,
                    'caption': caption,
                    'parse_mode': 'HTML'
                }
                requests.post(url, data=data, files=files, timeout=10)
        except Exception as e:
            print(f"[Telegram] Error sending photo: {e}")
