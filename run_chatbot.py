#!/usr/bin/env python3

import sys
from app.chatbot import ChatBot


class ChatBotRunner:
    def __init__(self):
        try:
            self.bot = ChatBot()
            print("🤖 CodeBot ready to work!")
            print("💡 Tip: Type 'quit', 'exit' or 'bye' to stop")
            print("🔧 Type 'help' for available commands")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            sys.exit(1)

    def show_help(self):
        print("\n📋 Available commands:")
        print("  help    - Show this help message")
        print("  test    - Run connection test")
        print("  clear   - Clear screen")
        print("  quit    - Exit chatbot")
        print("  exit    - Exit chatbot")
        print("  bye     - Exit chatbot")
        print("\n💬 Just type your question to chat with CodeBot!")

    def test_connection(self):
        print("\n🔍 Testing connection...")
        success = self.bot.test_connection()
        if success:
            print("✅ Connection successful!")
        else:
            print("❌ Connection failed!")

    def clear_screen(self):
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        print("🤖 CodeBot ready to work!")
        print("💡 Tip: Type 'quit', 'exit' or 'bye' to stop")
        print("🔧 Type 'help' for available commands")
        print("-" * 50)

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

                if user_input.lower() == 'help':
                    self.show_help()
                    continue

                if user_input.lower() == 'test':
                    self.test_connection()
                    continue

                if user_input.lower() == 'clear':
                    self.clear_screen()
                    continue

                print("🤖 CodeBot: 🤔 Thinking...")

                response = self.bot.get_response(user_input)

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
