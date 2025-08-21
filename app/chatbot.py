import os
from typing import Optional
import openai
from dotenv import load_dotenv
import toml


load_dotenv()


class ChatBot:
    def __init__(self, config_path: str = "config.toml"):
        self.config = toml.load(config_path)

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key in environment variables")

        self.client = openai.OpenAI(api_key=self.api_key)

        self.model_name = self.config["model"]["name"]
        self.temperature = self.config["model"]["temperature"]
        self.max_tokens = self.config["model"]["max_tokens"]
        self.system_prompt = self.config["system"]["preprompt"]

        print(f"✅ ChatBot initialized with model: {self.model_name}")

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
            print(f"❌ Error communicating with OpenAI: {e}")
            return None

    def start_chatting(self):
        print("🤖 CodeBot ready to work!")
        print("💡 Tip: Type 'q' to stop")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() == 'q':
                    print("👋 See you later!")
                    break

                if not user_input:
                    print("⚠️ Enter a question or command!")
                    continue

                print("🤖 CodeBot: 🤔 Thinking...")
                response = self.get_response(user_input)

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
