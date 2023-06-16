
import logging
from telegram import Update,Bot
from telegram.ext import ContextTypes

# Define function to initialize bot and get message object
def get_message_data(update: Update) -> tuple:
    TOKEN = os.getenv("API_BOT")
    bot = Bot(TOKEN)
    message = update.message
    user_full_name = f"{message.chat.first_name} {message.chat.last_name}"
    return bot, message, user_full_name

# Define function to handle the /start command   <-- START COMMAND-->
async def handle_start_command(update: Update, context: Context) -> None:
    bot, message, user_full_name = get_message_data(update)
    logging.info(f"/start command pressed by {user_full_name}")
    welcome_text = f"Hello {user_full_name},You are Happy"
    await bot.send_message(chat_id=message.chat_id, text=welcome_text)