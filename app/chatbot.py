import os
from typing import Optional
import openai
from dotenv import load_dotenv
import toml


load_dotenv()


class ChatBot:
    def __init__(self, config_path: str = "../config.toml"):
        self.config = toml.load(config_path)

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Brak klucza API OpenAI w zmiennych środowiskowych")

        self.client = openai.OpenAI(api_key=self.api_key)

        self.model_name = self.config["model"]["name"]
        self.temperature = self.config["model"]["temperature"]
        self.max_tokens = self.config["model"]["max_tokens"]
        self.system_prompt = self.config["system"]["preprompt"]

        print(f"✅ ChatBot zainicjalizowany z modelem: {self.model_name}")

    def get_response(self, user_message: str) -> Optional[str]:

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ Błąd podczas komunikacji z OpenAI: {e}")
            return None

    def test_connection(self) -> bool:
        test_message = "Powiedz 'Połączenie działa!' jeśli otrzymujesz tę wiadomość."

        response = self.get_response(test_message)

        if response:
            print("✅ Test połączenia zakończony sukcesem!")
            print(f"📝 Odpowiedź: {response}")
            return True
        else:
            print("❌ Test połączenia nieudany!")
            return False


def test_chatbot():
    try:
        bot = ChatBot()
        bot.test_connection()

        print("\n" + "=" * 50)
        print("🧪 TEST PODSTAWOWEJ FUNKCJONALNOŚCI")
        print("=" * 50)

        test_question = "Wyjaśnij mi krótko czym jest Python?"
        print(f"❓ Pytanie: {test_question}")

        response = bot.get_response(test_question)
        if response:
            print(f"🤖 Odpowiedź: {response}")
        else:
            print("❌ Brak odpowiedzi")

    except Exception as e:
        print(f"❌ Błąd podczas testowania: {e}")


if __name__ == "__main__":
    test_chatbot()
