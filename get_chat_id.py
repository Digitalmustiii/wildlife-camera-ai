from telegram_bot import get_chat_id

bot_token = "8580006006:AAFKwzyb9pftdDKyHFRX-HBedt1d_nOm1zU"

print("Getting your chat ID...")
print("⚠️  Make sure you sent /start to @my_wildlife_bot first!\n")

chat_id = get_chat_id(bot_token)

if chat_id:
    print(f"✅ Your Chat ID: {chat_id}")
    print(f"\nSave these credentials:")
    print(f"Bot Token: {bot_token}")
    print(f"Chat ID: {chat_id}")
else:
    print("❌ No messages found. Send /start to your bot first!")