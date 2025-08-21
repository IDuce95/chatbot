import sys
from app.chatbot import ChatBot


def main():
    print("🚀 Starting CodeBot")
    print("=" * 50)

    try:
        chatbot = ChatBot()
        chatbot.start_chatting()
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
