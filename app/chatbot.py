import os
from typing import Optional, List, Dict
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
        self.conversation_history: List[Dict[str, str]] = []

        print(f"\nCodeBot initialized with model: {self.config['model']['name']}")

    def get_response(self, user_message: str) -> Optional[str]:

        try:
            messages = [{"role": "system", "content": self.config["system"]["preprompt"]}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_message})

            response = self.client.chat.completions.create(
                model=self.config["model"]["name"],
                messages=messages,
                temperature=self.config["model"]["temperature"],
                max_tokens=self.config["model"]["max_tokens"],
            )

            assistant_response = response.choices[0].message.content

            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})

            return assistant_response

        except Exception as e:
            print(f"Error communicating with OpenAI: {e}")
            return None

    def start_chatting(self):
        print("Type 'q' to stop")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() == 'q':
                    print("🤖 CodeBot: See you later!")
                    break

                if not user_input:
                    print("⚠️ Enter a question or command!")
                    continue

                print("🤖 CodeBot: thinking...")
                response = self.get_response(user_input)

                if response:
                    print(f"🤖 CodeBot: {response}")
                else:
                    print("Failed to get response. Try again!")

            except KeyboardInterrupt:
                print("\n🤖 CodeBot: See you later!")
                break
            except Exception as e:
                print(f"Error occurred: {e}")
