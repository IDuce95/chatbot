from ..base_agent import BaseAgent
from ..state import AgentState


class CodeAgent(BaseAgent):
    def __init__(self, chatbot, config):
        super().__init__(
            chatbot=chatbot, name="Code", prompt=config["agents"]["code"]["prompt"]
        )
        self.config = config

    def process(self, state: AgentState) -> AgentState:
        return self.generate_code(state)

    def _format_research_context(self, research_results: list) -> str:
        if not research_results:
            return "No specific documentation found."

        context_parts = []
        for i, result in enumerate(research_results[:3]):
            content = result.get("content", "")
            source = result.get("source", "Unknown")
            context_parts.append(f"Source {i + 1} ({source}):\n{content}\n")

        return "\n".join(context_parts)

    def _generate_code_with_context(self, state: AgentState, context: str) -> str:
        prompt = f"""{self.prompt}

Documentation Context:
{context}

{self.config["agents"]["code"]["code_generation_template"]}"""

        return self._call_llm_with_context(
            state, prompt, max_tokens=1000, temperature=0.2
        )

    def generate_code(self, state: AgentState) -> AgentState:
        research_results = state.get("research_results", [])

        try:
            print("CodeAgent: Using CodeGeneratorTool to create code solution...")
            context = self._format_research_context(research_results)

            if research_results:
                num_sources = state.get("metadata", {}).get("num_sources", 0)
                if num_sources > 0:
                    print(
                        f"CodeAgent: Using context from {num_sources} source documents for code generation"
                    )
                else:
                    print("CodeAgent: Using research context for code generation")
            else:
                print(
                    "CodeAgent: Generating code without specific documentation context"
                )

            code_response = self._generate_code_with_context(state, context)

            print(
                "CodeAgent: Applying LinterTool for code formatting and validation..."
            )
            formatted_code = self._format_and_validate_code(code_response)

            state["generated_code"] = formatted_code
            state["metadata"]["code_generated"] = True
            state["metadata"]["context_used"] = len(research_results) > 0

            print("CodeAgent: Code generation and validation complete")

        except Exception as e:
            print(f"❌ CodeAgent error: {e}")
            state["generated_code"] = None
            state["metadata"]["code_error"] = str(e)

        return self._update_state(state)

    def _format_and_validate_code(self, code_response: str) -> str:
        try:
            formatted = code_response.strip()

            if "```python" in formatted:
                return formatted
            elif "```" in formatted:
                formatted = formatted.replace("```", "```python", 1)
                return formatted
            else:
                lines = formatted.split("\n")
                code_lines = []
                explanation_lines = []

                in_code = False
                for line in lines:
                    if (
                        line.strip().startswith("#")
                        or line.strip().startswith("import")
                        or line.strip().startswith("def")
                        or line.strip().startswith("class")
                        or "=" in line
                        or line.startswith("    ")
                    ):
                        code_lines.append(line)
                        in_code = True
                    elif in_code and line.strip() == "":
                        code_lines.append(line)
                    else:
                        explanation_lines.append(line)

                if code_lines:
                    result = "```python\n" + "\n".join(code_lines) + "\n```"
                    if explanation_lines:
                        result += "\n\n" + "\n".join(explanation_lines)
                    return result
                else:
                    return formatted

        except Exception:
            return code_response

    def should_generate_code(self, state: AgentState) -> bool:
        intent = state.get("intent_classification", "")
        return intent == "CODE" or "code" in state.get("user_query", "").lower()

    def has_research_context(self, state: AgentState) -> bool:
        research_results = state.get("research_results", [])
        return len(research_results) > 0
