
from telegram.ext import CallbackContext,Application,CommandHandler
from telegram import Update,Bot

from handler import handle_start_command

# Define an error handler function
def error_handler(update: Update, context: CallbackContext):
    # Log the exception details
    print(f"Exception occurred: {context.error}")

if __name__ == '__main__':
    try:
         # my code
        application = Application.builder().token(token="5733145281:AAH9v6FcGXykasJp207_tOI9_I81SLtnW9k").build()
        application.add_error_handler(error_handler)

        application.add_handler(CommandHandler('start',handle_start_command))
        # application.add_handler(CommandHandler('stop',handle_stop_command))
        
        # Start the bot
        application.run_polling(1.0)
    except Exception as e:
        # Handle other exceptions
        print("An error occurred:", e)
