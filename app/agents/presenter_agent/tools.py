import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class TextFormatterTool:
    def __init__(self, agent_ref):
        self.agent = agent_ref

    def process(self, content: str, metadata: Dict[str, Any]) -> str:
        try:
            return self._format_text(content, metadata)
        except Exception as e:
            logger.error(f"Text formatting error: {e}")
            return content

    def _format_text(self, content: str, metadata: Dict[str, Any]) -> str:
        if not content.strip():
            return "I don't have a specific answer for that question."

        if len(content) > 5000:
            content = content[:4500] + "\n\n*[Response truncated for readability]*"

        agent_type = metadata.get("agent_type", "")

        if agent_type == "code":
            return self._format_code_response(content)
        elif agent_type == "research":
            return self._format_research_response(content, metadata)
        elif agent_type == "conversation":
            return self._format_conversation_response(content)

        return content

    def _format_code_response(self, content: str) -> str:
        if "def " in content and "```" not in content:
            lines = content.split('\n')
            code_started = False
            formatted_lines = []

            for line in lines:
                if any(keyword in line for keyword in ["def ", "class ", "import ", "from "]):
                    if not code_started:
                        formatted_lines.append("```python")
                        code_started = True
                    formatted_lines.append(line)
                elif code_started and (line.strip() == "" or line.startswith("    ")):
                    formatted_lines.append(line)
                else:
                    if code_started:
                        formatted_lines.append("```")
                        code_started = False
                    formatted_lines.append(line)

            if code_started:
                formatted_lines.append("```")

            return '\n'.join(formatted_lines)

        return content

    def _format_research_response(self, content: str, metadata: Dict[str, Any]) -> str:
        response = content

        sources = metadata.get("sources", [])
        if sources:
            response += "\n\n**Sources:**\n"
            for i, source in enumerate(sources[:3], 1):
                response += f"{i}. {source}\n"

        return response

    def _format_conversation_response(self, content: str) -> str:
        if len(content.split()) < 3:
            return f"✨ {content}"
        return content
