from chatbot import ChatBot
import time


def comprehensive_test():
    """Kompleksowy test funkcjonalności ChatBota"""
    try:
        print("🧪 KOMPLEKSOWY TEST CHATBOTA")
        print("=" * 60)

        bot = ChatBot()

        print("\n1️⃣ TEST POŁĄCZENIA Z API")
        print("-" * 30)
        connection_ok = bot.test_connection()

        if not connection_ok:
            print("❌ Test zakończony - brak połączenia z API")
            return False

        print("\n2️⃣ TEST WIEDZY O PYTHON")
        print("-" * 30)
        python_questions = [
            "Jak działa list comprehension w Python?",
            "Co to są dekoratory w Python?",
            "Wyjaśnij różnicę między metodami klasy a metodami statycznymi",
        ]

        for i, question in enumerate(python_questions, 1):
            print(f"\n📝 Pytanie {i}: {question}")
            response = bot.get_response(question)
            if response and len(response) > 50:
                print("✅ Odpowiedź: OK (szczegółowa odpowiedź)")
            else:
                print("❌ Odpowiedź: Zbyt krótka lub brak")

        print("\n3️⃣ TEST WIEDZY O BIBLIOTEKACH")
        print("-" * 30)
        library_questions = [
            "Jak używać pandas do wczytania pliku CSV?",
            "Pokaż przykład użycia numpy do operacji na macierzach",
            "Jak stworzyć prostą aplikację Flask?",
        ]

        for i, question in enumerate(library_questions, 1):
            print(f"\n📝 Pytanie {i}: {question}")
            response = bot.get_response(question)
            if response and ("import" in response or "def" in response):
                print("✅ Odpowiedź: OK (zawiera przykłady kodu)")
            else:
                print("❌ Odpowiedź: Brak przykładów kodu")

        print("\n4️⃣ TEST PYTAŃ OGÓLNYCH")
        print("-" * 30)
        general_questions = [
            "Jakie są zalety pracy zdalnej?",
            "Opowiedz o historii komputerów",
        ]

        for i, question in enumerate(general_questions, 1):
            print(f"\n📝 Pytanie {i}: {question}")
            response = bot.get_response(question)
            if response and len(response) > 100:
                print("✅ Odpowiedź: OK (odpowiada na pytania spoza programowania)")
            else:
                print("❌ Odpowiedź: Za krótka")

        print("\n5️⃣ TEST WYDAJNOŚCI")
        print("-" * 30)
        start_time = time.time()
        _ = bot.get_response("Co to jest Python?")
        end_time = time.time()

        response_time = end_time - start_time
        print(f"⏱️ Czas odpowiedzi: {response_time:.2f} sekund")

        if response_time < 10:
            print("✅ Wydajność: OK (< 10 sekund)")
        else:
            print("⚠️ Wydajność: Wolna (> 10 sekund)")

        print("\n6️⃣ TEST KONFIGURACJI")
        print("-" * 30)
        print(f"🤖 Model: {bot.model_name}")
        print(f"🌡️ Temperature: {bot.temperature}")
        print(f"📏 Max tokens: {bot.max_tokens}")
        print(f"💬 System prompt: {'Ustawiony' if bot.system_prompt else 'Brak'}")

        print("\n" + "=" * 60)
        print("✅ WSZYSTKIE TESTY ZAKOŃCZONE!")
        print("🎯 ChatBot jest gotowy do użycia w Fazie 3")
        return True

    except Exception as e:
        print(f"❌ Błąd podczas testów: {e}")
        return False


def performance_test():
    """Test wydajności z wieloma zapytaniami"""
    try:
        print("\n🚀 TEST WYDAJNOŚCI - SERIA ZAPYTAŃ")
        print("=" * 50)

        bot = ChatBot()

        questions = [
            "Co to jest Python?",
            "Jak utworzyć funkcję?",
            "Co to są listy?",
            "Wyjaśnij pętle for",
            "Co to jest pandas?"
        ]

        total_time = 0
        successful_responses = 0

        for i, question in enumerate(questions, 1):
            start = time.time()
            response = bot.get_response(question)
            end = time.time()

            duration = end - start
            total_time += duration

            if response:
                successful_responses += 1
                status = "✅"
            else:
                status = "❌"

            print(f"{status} Pytanie {i}: {duration:.2f}s")

        avg_time = total_time / len(questions)
        success_rate = (successful_responses / len(questions)) * 100

        print("\n📊 WYNIKI:")
        print(f"⏱️ Średni czas odpowiedzi: {avg_time:.2f}s")
        print(f"✅ Wskaźnik sukcesu: {success_rate:.1f}%")
        print(f"🔄 Łączny czas: {total_time:.2f}s")

    except Exception as e:
        print(f"❌ Błąd podczas testu wydajności: {e}")


if __name__ == "__main__":
    print("🎯 Wybierz test:")
    print("1. Kompleksowy test funkcjonalności")
    print("2. Test wydajności")
    print("3. Oba testy")

    choice = input("\nWybór (1/2/3): ").strip()

    if choice == "1":
        comprehensive_test()
    elif choice == "2":
        performance_test()
    elif choice == "3":
        comprehensive_test()
        performance_test()
    else:
        print("❌ Nieprawidłowy wybór!")
