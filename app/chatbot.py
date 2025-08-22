import os
from typing import Dict, List, Optional

import openai
import toml
from dotenv import load_dotenv
from rag_manager import RAGManager

load_dotenv()


class ChatBot:
    def __init__(self, config_path: str = "config.toml", use_rag: bool = True):
        self.config = toml.load(config_path)

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key in environment variables")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.conversation_history: List[Dict[str, str]] = []
        self.model_info = f"CodeBot initialized with model: {self.config['model']['name']}"
        self.last_rag_used = False

        self.use_rag = use_rag
        self.rag_manager = None
        if use_rag:
            try:
                self.rag_manager = RAGManager()
                self.model_info += " (with RAG knowledge base)"
            except Exception as e:
                print(f"Warning: Could not initialize RAG manager: {e}")
                self.use_rag = False

    def get_response(self, user_message: str) -> Optional[str]:
        try:
            context = ""
            rag_used = False

            if self.use_rag and self.rag_manager:
                try:
                    context, rag_used = self.rag_manager.get_context_with_relevance(user_message, k=3)
                except Exception as e:
                    print(f"Warning: RAG search failed: {e}")
                    context = ""
                    rag_used = False

            system_content = self.config["system"]["preprompt"]
            if context:
                system_content += f"\n\nRelevant context from knowledge base:\n{context}"

            messages = [{"role": "system", "content": system_content}]
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

            self.last_rag_used = rag_used

            return assistant_response

        except Exception as e:
            print(f"Error communicating with OpenAI: {e}")
            return None

    def clear_history(self):
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()

    def get_model_info(self) -> str:
        return self.model_info

    def start_chatting(self):
        print(self.model_info)
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
