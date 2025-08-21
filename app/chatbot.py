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
