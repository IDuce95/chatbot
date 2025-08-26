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

        self.agent_graph = None
        self.use_agents = False
        try:
            from agents.agent_graph import AgentGraph
            self.agent_graph = AgentGraph(self, self.config)
            self.use_agents = True
            self.model_info += " + Multi-Agent System"
            print("✅ Multi-Agent System initialized successfully!")
        except Exception as e:
            print(f"Warning: Could not initialize Agent Graph: {e}")
            self.use_agents = False

    def get_response(self, user_message: str) -> Optional[str]:
        try:
            if self.use_agents and self.agent_graph:
                return self.get_response_with_agents(user_message)
            else:
                return self.get_response_legacy(user_message)
        except Exception as e:
            print(f"Error in get_response: {e}")
            raise

    def get_response_with_agents(self, user_message: str) -> Optional[str]:
        print(f"🔧 DEBUG: get_response_with_agents STARTED for: {user_message[:30]}...")
        try:
            import time
            start_time = time.time()

            result = self.agent_graph.process_query(user_message, self.conversation_history)
            print("🔧 DEBUG: agent_graph.process_query COMPLETED")

            response_text = result.get("response", "")
            agents_used = result.get("agents_used", [])
            intent = result.get("intent", "")
            quality_score = result.get("quality_score", 0.0)
            research_results = result.get("research_results", [])

            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            end_time = time.time()
            response_time = end_time - start_time

            self.last_rag_used = "research" in agents_used

            if self.rag_manager and hasattr(self.rag_manager, 'metrics'):
                try:
                    self.rag_manager.metrics.log_interaction(
                        query=user_message,
                        retrieved_docs=research_results,
                        response=response_text,
                        context=f"Agents: {', '.join(agents_used)} | Intent: {intent}",
                        response_time=response_time,
                        rag_used=self.last_rag_used
                    )
                except Exception as e:
                    print(f"Warning: Failed to log metrics: {e}")

            print(f"🤖 Agents used: {agents_used} | Intent: {intent} | Quality: {quality_score:.2f}")

            return response_text

        except Exception as e:
            print(f"Agent system error: {e}")
            return self.get_response_legacy(user_message)

    def get_response_legacy(self, user_message: str) -> Optional[str]:
        try:
            import time
            start_time = time.time()

            context = ""
            rag_used = False
            retrieved_docs_info = []

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
            end_time = time.time()
            response_time = end_time - start_time

            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})

            self.last_rag_used = rag_used

            if self.use_rag and self.rag_manager:
                self.rag_manager.metrics.log_interaction(
                    query=user_message,
                    retrieved_docs=retrieved_docs_info,
                    response=assistant_response,
                    context=context,
                    response_time=response_time,
                    rag_used=rag_used
                )

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

    def get_metrics_summary(self) -> dict:
        if self.use_rag and self.rag_manager:
            return self.rag_manager.get_metrics_summary()
        return {"status": "RAG not enabled"}

    def export_metrics(self, filepath: str = "chatbot_metrics.json"):
        if self.use_rag and self.rag_manager:
            self.rag_manager.export_metrics(filepath)
        else:
            print("RAG not enabled, no metrics to export")

    def clear_metrics(self):
        if self.use_rag and self.rag_manager:
            self.rag_manager.clear_metrics()
        else:
            print("RAG not enabled, no metrics to clear")

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
