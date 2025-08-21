from chatbot import ChatBot
import time


def comprehensive_test():
    try:
        print("🧪 COMPREHENSIVE CHATBOT TEST")
        print("=" * 60)

        bot = ChatBot()

        print("\n1️⃣ API CONNECTION TEST")
        print("-" * 30)
        connection_ok = bot.test_connection()

        if not connection_ok:
            print("❌ Test ended - no API connection")
            return False

        print("\n2️⃣ PYTHON KNOWLEDGE TEST")
        print("-" * 30)
        python_questions = [
            "How does list comprehension work in Python?",
            "What are decorators in Python?",
            "Explain the difference between class methods and static methods",
        ]

        for i, question in enumerate(python_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            response = bot.get_response(question)
            if response and len(response) > 50:
                print("✅ Response: OK (detailed response)")
            else:
                print("❌ Response: Too short or missing")

        print("\n3️⃣ LIBRARIES KNOWLEDGE TEST")
        print("-" * 30)
        library_questions = [
            "How to use pandas to read a CSV file?",
            "Show an example of using numpy for matrix operations",
            "How to create a simple Flask application?",
        ]

        for i, question in enumerate(library_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            response = bot.get_response(question)
            if response and ("import" in response or "def" in response):
                print("✅ Response: OK (contains code examples)")
            else:
                print("❌ Response: Missing code examples")

        print("\n4️⃣ GENERAL QUESTIONS TEST")
        print("-" * 30)
        general_questions = [
            "What are the advantages of remote work?",
            "Tell me about computer history",
        ]

        for i, question in enumerate(general_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            response = bot.get_response(question)
            if response and len(response) > 100:
                print("✅ Response: OK (answers non-programming questions)")
            else:
                print("❌ Response: Too short")

        print("\n5️⃣ PERFORMANCE TEST")
        print("-" * 30)
        start_time = time.time()
        _ = bot.get_response("What is Python?")
        end_time = time.time()

        response_time = end_time - start_time
        print(f"⏱️ Response time: {response_time:.2f} seconds")

        if response_time < 10:
            print("✅ Performance: OK (< 10 seconds)")
        else:
            print("⚠️ Performance: Slow (> 10 seconds)")

        print("\n6️⃣ CONFIGURATION TEST")
        print("-" * 30)
        print(f"🤖 Model: {bot.model_name}")
        print(f"🌡️ Temperature: {bot.temperature}")
        print(f"📏 Max tokens: {bot.max_tokens}")
        print(f"💬 System prompt: {'Set' if bot.system_prompt else 'Missing'}")

        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED!")
        print("🎯 ChatBot is ready for Phase 3")
        return True

    except Exception as e:
        print(f"❌ Error during tests: {e}")
        return False


def performance_test():
    try:
        print("\n🚀 PERFORMANCE TEST - SERIES OF QUERIES")
        print("=" * 50)

        bot = ChatBot()

        questions = [
            "What is Python?",
            "How to create a function?",
            "What are lists?",
            "Explain for loops",
            "What is pandas?"
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

            print(f"{status} Question {i}: {duration:.2f}s")

        avg_time = total_time / len(questions)
        success_rate = (successful_responses / len(questions)) * 100

        print("\n📊 RESULTS:")
        print(f"⏱️ Average response time: {avg_time:.2f}s")
        print(f"✅ Success rate: {success_rate:.1f}%")
        print(f"🔄 Total time: {total_time:.2f}s")

    except Exception as e:
        print(f"❌ Error during performance test: {e}")


if __name__ == "__main__":
    print("🎯 Choose test:")
    print("1. Comprehensive functionality test")
    print("2. Performance test")
    print("3. Both tests")

    choice = input("\nChoice (1/2/3): ").strip()

    if choice == "1":
        comprehensive_test()
    elif choice == "2":
        performance_test()
    elif choice == "3":
        comprehensive_test()
        performance_test()
    else:
        print("❌ Invalid choice!")
