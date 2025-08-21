from chatbot import ChatBot


class SimpleChatBot:
    def __init__(self):
        try:
            self.bot = ChatBot()
            print("🤖 CodeBot gotowy do działania!")
            print("💡 Tip: Wpisz 'quit', 'exit' lub 'bye' aby zakończyć")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Błąd inicjalizacji: {e}")
            raise

    def start_chat(self):
        while True:
            try:
                user_input = input("\n👤 Ty: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("👋 Do zobaczenia!")
                    break

                if not user_input:
                    print("⚠️  Wprowadź pytanie lub komendę!")
                    continue

                print("🤖 CodeBot: 🤔 Myślę...")

                response = self.bot.get_response(user_input)

                if response:
                    print(f"🤖 CodeBot: {response}")
                else:
                    print("❌ Nie udało się uzyskać odpowiedzi. Spróbuj ponownie!")

            except KeyboardInterrupt:
                print("\n\n👋 Do zobaczenia!")
                break
            except Exception as e:
                print(f"❌ Wystąpił błąd: {e}")
                print("🔄 Spróbuj ponownie...")


def demo_questions():
    try:
        bot = ChatBot()

        demo_questions_list = [
            "Czym jest Python?",
            "Jak utworzyć listę w Python?",
            "Co to jest pandas?",
            "Jaka jest różnica między listą a tuple?",
            "Co sądzisz o pogodzie?"
        ]

        print("🚀 DEMO - Przykładowe pytania")
        print("=" * 50)

        for i, question in enumerate(demo_questions_list, 1):
            print(f"\n📝 Pytanie {i}: {question}")
            print("🤖 Odpowiedź:", end=" ")

            response = bot.get_response(question)
            if response:
                short_response = response[:200] + "..." if len(response) > 200 else response
                print(short_response)
            else:
                print("❌ Brak odpowiedzi")

            print("-" * 30)

    except Exception as e:
        print(f"❌ Błąd podczas demo: {e}")


if __name__ == "__main__":
    print("🎯 Wybierz opcję:")
    print("1. Interaktywny chat")
    print("2. Demo z przykładowymi pytaniami")

    choice = input("\nWybór (1/2): ").strip()

    if choice == "1":
        simple_bot = SimpleChatBot()
        simple_bot.start_chat()
    elif choice == "2":
        demo_questions()
    else:
        print("❌ Nieprawidłowy wybór!")
