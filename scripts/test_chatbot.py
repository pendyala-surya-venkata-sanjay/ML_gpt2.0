import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.chatbot.conversation_manager import ConversationManager


def run_test():

    print("\n===== CHATBOT TEST STARTED =====\n")

    bot = ConversationManager()

    # simulate dataset upload
    bot.dataset_path = "datasets/sample_dataset.csv"

    print("USER: train model")
    print("BOT:", bot.handle_message("train model"))

    print("\n--------------------------\n")

    print("USER: explain random forest")
    print("BOT:", bot.handle_message("explain random forest"))

    print("\n--------------------------\n")

    print("USER: what is feature scaling")
    print("BOT:", bot.handle_message("what is feature scaling"))

    print("\n===== CHATBOT TEST FINISHED =====\n")


if __name__ == "__main__":
    run_test()