
import requests
from pathlib import Path
from typing import Optional
from datetime import datetime


class TelegramBot:
    """Telegram bot for wildlife notifications"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram bot
        
        Args:
            bot_token: Bot token from @BotFather
            chat_id: Your Telegram chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self._validate_connection()
    
    def _validate_connection(self):
        """Test bot connection"""
        try:
            response = requests.get(f"{self.base_url}/getMe", timeout=5)
            if response.status_code == 200:
                bot_info = response.json()
                print(f"✅ Telegram bot connected: @{bot_info['result']['username']}")
            else:
                print(f"⚠️  Telegram bot auth failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️  Telegram connection error: {e}")
    
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send text message
        
        Args:
            text: Message text (supports HTML/Markdown)
            parse_mode: 'HTML' or 'Markdown'
            
        Returns:
            True if sent successfully
        """
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                data={
                    'chat_id': self.chat_id,
                    'text': text,
                    'parse_mode': parse_mode
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"⚠️  Telegram send error: {e}")
            return False
    
    def send_photo(self, photo_path: str, caption: Optional[str] = None) -> bool:
        """
        Send photo with optional caption
        
        Args:
            photo_path: Path to image file
            caption: Photo caption (supports HTML)
            
        Returns:
            True if sent successfully
        """
        if not Path(photo_path).exists():
            print(f"⚠️  Photo not found: {photo_path}")
            return False
        
        try:
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.chat_id,
                    'parse_mode': 'HTML'
                }
                if caption:
                    data['caption'] = caption
                
                response = requests.post(
                    f"{self.base_url}/sendPhoto",
                    files=files,
                    data=data,
                    timeout=15
                )
                return response.status_code == 200
        except Exception as e:
            print(f"⚠️  Photo send error: {e}")
            return False
    
    def send_alert(self, event, snapshot_path: Optional[str] = None) -> bool:
        """
        Send wildlife alert notification
        
        Args:
            event: AlertEvent object
            snapshot_path: Path to snapshot image
            
        Returns:
            True if sent successfully
        """
        # Format message
        rarity_emoji = {
            'common': '🟢',
            'interesting': '🟡',
            'rare': '🔴'
        }
        emoji = rarity_emoji.get(event.rarity, '⚪')
        
        message = f"""
{emoji} <b>Wildlife Alert: {event.species.upper()}</b>

📊 Confidence: {event.confidence:.0%}
⭐ Rarity: {event.rarity.title()}
🕐 Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """.strip()
        
        # Send with photo if available
        if snapshot_path and Path(snapshot_path).exists():
            return self.send_photo(snapshot_path, caption=message)
        else:
            return self.send_message(message)
    
    def send_daily_summary(self, stats: dict) -> bool:
        """
        Send daily activity summary
        
        Args:
            stats: Dictionary with detection statistics
            
        Returns:
            True if sent successfully
        """
        message = f"""
📊 <b>Daily Wildlife Summary</b>

🔢 Total Detections: {stats.get('total_detections', 0)}
🦌 Unique Species: {stats.get('unique_species', 0)}
🔴 Rare Sightings: {stats.get('rare_count', 0)}

<b>Top Species:</b>
"""
        
        species_breakdown = stats.get('species_breakdown', {})
        for species, count in sorted(species_breakdown.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
            message += f"• {species.title()}: {count}\n"
        
        return self.send_message(message.strip())


def create_telegram_alert_handler(bot_token: str, chat_id: str):
    """
    Create Telegram alert handler for AlertManager
    
    Args:
        bot_token: Telegram bot token
        chat_id: Your chat ID
        
    Returns:
        Handler function
    """
    bot = TelegramBot(bot_token, chat_id)
    
    def handler(event):
        """Handle alert event"""
        snapshot = event.snapshot_path if hasattr(event, 'snapshot_path') else None
        bot.send_alert(event, snapshot)
    
    return handler


# Utility: Get your Telegram chat ID
def get_chat_id(bot_token: str) -> Optional[str]:
    """
    Get your chat ID (send /start to your bot first)
    
    Args:
        bot_token: Bot token from @BotFather
        
    Returns:
        Chat ID or None
    """
    try:
        url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data['result']:
                chat_id = data['result'][-1]['message']['chat']['id']
                return str(chat_id)
    except Exception as e:
        print(f"Error getting chat ID: {e}")
    
    return None


# Setup guide helper
def print_setup_guide():
    """Print Telegram bot setup instructions"""
    guide = """
╔════════════════════════════════════════════════════════════╗
║            TELEGRAM BOT SETUP GUIDE                        ║
╚════════════════════════════════════════════════════════════╝

Step 1: Create Bot
  1. Open Telegram and search for @BotFather
  2. Send: /newbot
  3. Choose a name (e.g., "My Wildlife Camera")
  4. Choose a username (e.g., "my_wildlife_bot")
  5. Copy the bot token (looks like: 123456:ABC-DEF1234...)

Step 2: Get Your Chat ID
  1. Search for your bot in Telegram
  2. Send: /start
  3. Run this Python code:
  
     from telegram_bot import get_chat_id
     chat_id = get_chat_id('YOUR_BOT_TOKEN')
     print(f"Your chat ID: {chat_id}")

Step 3: Configure
  Update your config.json or code:
  
  config.alert.telegram_enabled = True
  config.alert.telegram_bot_token = "YOUR_BOT_TOKEN"
  config.alert.telegram_chat_id = "YOUR_CHAT_ID"

Step 4: Test
  from telegram_bot import TelegramBot
  
  bot = TelegramBot('YOUR_TOKEN', 'YOUR_CHAT_ID')
  bot.send_message("🎉 Wildlife Camera connected!")

Done! You'll now receive alerts on your phone.

╚════════════════════════════════════════════════════════════╝
    """
    print(guide)


if __name__ == '__main__':
    # Print setup guide when run directly
    print_setup_guide()