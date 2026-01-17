from telegram_bot import TelegramBot

bot_token = "8580006006:AAFKwzyb9pftdDKyHFRX-HBedt1d_nOm1zU"
chat_id = "7256925056"

bot = TelegramBot(bot_token, chat_id)

# Test with detailed response
print("Sending message...")
success = bot.send_message("🎉 Test message from Wildlife Camera!")

if success:
    print("✅ Message sent successfully!")
else:
    print("❌ Message failed to send")
    
# Try getting bot info
import requests
response = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe")
print(f"\nBot info: {response.json()}")

# Check if message was received
response = requests.get(f"https://api.telegram.org/bot{bot_token}/getUpdates")
print(f"\nRecent updates: {response.json()}")