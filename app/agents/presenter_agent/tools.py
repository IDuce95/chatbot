import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from ...pydantic_models import ResponseFormat

logger = logging.getLogger(__name__)


class PresentationTool(ABC):
    def __init__(self, agent_ref):
        self.agent = agent_ref

    @abstractmethod
    def process(self, content: str, metadata: Dict[str, Any]) -> ResponseFormat:
        pass


class TextFormatterTool(PresentationTool):
    def process(self, content: str, metadata: Dict[str, Any]) -> ResponseFormat:
        try:
            formatted_content = self._format_text(content, metadata)
            return ResponseFormat(
                content=formatted_content,
                metadata=metadata,
                format_type="text"
            )
        except Exception as e:
            logger.error(f"Text formatting error: {e}")
            return ResponseFormat(content=content, metadata=metadata)

    def _format_text(self, content: str, metadata: Dict[str, Any]) -> str:
        if not content.strip():
            return "I don't have a specific answer for that question."

        if metadata.get("agent_type") == "code":
            return self._format_code_response(content)
        elif metadata.get("agent_type") == "research":
            return self._format_research_response(content, metadata)
        elif metadata.get("agent_type") == "conversation":
            return self._format_conversation_response(content)

        return content

    def _format_code_response(self, content: str) -> str:
        lines = content.split('\n')
        formatted_lines = []

        for line in lines:
            if line.strip().startswith('```'):
                formatted_lines.append(line)
            elif 'def ' in line or 'class ' in line:
                formatted_lines.append(f"**{line.strip()}**")
            else:
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _format_research_response(self, content: str, metadata: Dict[str, Any]) -> str:
        response = content

        if metadata.get("sources"):
            sources = metadata["sources"]
            if sources:
                response += "\n\n**Sources:**\n"
                for i, source in enumerate(sources[:3], 1):
                    response += f"{i}. {source}\n"

        return response

    def _format_conversation_response(self, content: str) -> str:
        if len(content.split()) < 3:
            return f"✨ {content}"
        return content


class MarkdownFormatterTool(PresentationTool):
    def process(self, content: str, metadata: Dict[str, Any]) -> ResponseFormat:
        try:
            markdown_content = self._convert_to_markdown(content, metadata)
            return ResponseFormat(
                content=markdown_content,
                metadata=metadata,
                format_type="markdown"
            )
        except Exception as e:
            logger.error(f"Markdown formatting error: {e}")
            return ResponseFormat(content=content, metadata=metadata)

    def _convert_to_markdown(self, content: str, metadata: Dict[str, Any]) -> str:
        lines = content.split('\n')
        markdown_lines = []

        for line in lines:
            if line.strip().startswith('#'):
                markdown_lines.append(line)
            elif line.strip().startswith('**') and line.strip().endswith('**'):
                markdown_lines.append(line)
            elif 'def ' in line or 'class ' in line:
                markdown_lines.append(f"### {line.strip()}")
            elif line.strip() and not line.startswith(' '):
                if len(line.split()) > 10:
                    markdown_lines.append(f"\n{line}\n")
                else:
                    markdown_lines.append(line)
            else:
                markdown_lines.append(line)

        return '\n'.join(markdown_lines)


class ResponseEnhancerTool(PresentationTool):
    def process(self, content: str, metadata: Dict[str, Any]) -> ResponseFormat:
        try:
            enhanced_content = self._enhance_response(content, metadata)
            return ResponseFormat(
                content=enhanced_content,
                metadata=metadata,
                format_type="enhanced"
            )
        except Exception as e:
            logger.error(f"Response enhancement error: {e}")
            return ResponseFormat(content=content, metadata=metadata)

    def _enhance_response(self, content: str, metadata: Dict[str, Any]) -> str:
        if not content or len(content.strip()) < 10:
            return "I need more information to provide a comprehensive answer."

        agent_type = metadata.get("agent_type", "")
        confidence = metadata.get("confidence", 0.0)

        enhanced = content

        if confidence < 0.3:
            enhanced = f"Based on available information: {enhanced}"
        elif confidence > 0.8:
            enhanced = f"✅ {enhanced}"

        if agent_type == "research" and metadata.get("source_count", 0) > 3:
            enhanced += "\n\n*This answer is based on multiple reliable sources.*"
        elif agent_type == "code" and "```" in content:
            enhanced += "\n\n*Code example provided above.*"

        return enhanced


class QualityControlTool(PresentationTool):
    def process(self, content: str, metadata: Dict[str, Any]) -> ResponseFormat:
        try:
            quality_check = self._check_quality(content, metadata)

            if not quality_check["is_valid"]:
                content = quality_check["fallback"]
                metadata["quality_issues"] = quality_check["issues"]

            return ResponseFormat(
                content=content,
                metadata=metadata,
                format_type="quality_checked"
            )
        except Exception as e:
            logger.error(f"Quality control error: {e}")
            return ResponseFormat(content=content, metadata=metadata)

    def _check_quality(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        issues = []

        if not content or len(content.strip()) < 5:
            issues.append("Content too short")

        if len(content) > 5000:
            issues.append("Content too long")

        if content.count('\n') > 100:
            issues.append("Too many line breaks")

        if metadata.get("agent_type") == "code" and "```" not in content and "def " in content:
            issues.append("Code not properly formatted")

        is_valid = len(issues) == 0

        fallback = content
        if not is_valid:
            if not content.strip():
                fallback = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            elif len(content) > 5000:
                fallback = content[:4500] + "\n\n*[Response truncated for readability]*"

        return {
            "is_valid": is_valid,
            "issues": issues,
            "fallback": fallback
        }


class MetadataEnricherTool(PresentationTool):
    def process(self, content: str, metadata: Dict[str, Any]) -> ResponseFormat:
        try:
            enriched_metadata = self._enrich_metadata(content, metadata)
            return ResponseFormat(
                content=content,
                metadata=enriched_metadata,
                format_type="metadata_enriched"
            )
        except Exception as e:
            logger.error(f"Metadata enrichment error: {e}")
            return ResponseFormat(content=content, metadata=metadata)

    def _enrich_metadata(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        enriched = metadata.copy()

        enriched["response_length"] = len(content)
        enriched["word_count"] = len(content.split())
        enriched["line_count"] = len(content.split('\n'))

        if "```" in content:
            enriched["contains_code"] = True
            enriched["code_blocks"] = content.count("```") // 2

        if "http" in content.lower():
            enriched["contains_links"] = True

        if any(word in content.lower() for word in ["error", "exception", "failed", "problem"]):
            enriched["contains_errors"] = True

        if any(word in content.lower() for word in ["example", "demo", "tutorial", "guide"]):
            enriched["educational_content"] = True

        return enriched
