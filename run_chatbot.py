#!/usr/bin/env python3

import sys
from app.chatbot import ChatBot


class ChatBotRunner:
    def __init__(self):
        try:
            self.chatbot = ChatBot()
            print("🤖 CodeBot ready to work!")
            print("💡 Tip: Type 'quit', 'exit' or 'bye' to stop")
            print("🔧 Type 'help' for available commands")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            sys.exit(1)

    def start_chat(self):
        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("👋 See you later!")
                    break

                if not user_input:
                    print("⚠️  Enter a question or command!")
                    continue

                print("🤖 CodeBot: 🤔 Thinking...")
                response = self.chatbot.get_response(user_input)

                if response:
                    print(f"🤖 CodeBot: {response}")
                else:
                    print("❌ Failed to get response. Try again!")

            except KeyboardInterrupt:
                print("\n\n👋 See you later!")
                break
            except Exception as e:
                print(f"❌ Error occurred: {e}")
                print("🔄 Try again...")


def main():
    print("🚀 Starting CodeBot - Programming Assistant")
    print("=" * 50)

    runner = ChatBotRunner()
    runner.start_chat()


if __name__ == "__main__":
    main()
