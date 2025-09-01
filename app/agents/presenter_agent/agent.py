import logging
from typing import Dict, Any, Optional
from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PresenterAgent(BaseAgent):
    def __init__(self, chatbot_ref, config):
        prompt = config.get("agents", {}).get("presenter", {}).get("prompt", "")
        super().__init__(chatbot_ref, "Presenter", prompt)
        self.config = config
        self.agent_type = "presenter"
        self._initialize_tools()

    def _initialize_tools(self):
        from .tools import (
            TextFormatterTool,
            MarkdownFormatterTool,
            ResponseEnhancerTool,
            QualityControlTool,
            MetadataEnricherTool
        )

        self.text_formatter = TextFormatterTool(self)
        self.markdown_formatter = MarkdownFormatterTool(self)
        self.response_enhancer = ResponseEnhancerTool(self)
        self.quality_controller = QualityControlTool(self)
        self.metadata_enricher = MetadataEnricherTool(self)

    def process(self, agent_response: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if metadata is None:
            metadata = {}

        try:
            metadata = self._enrich_metadata(agent_response, metadata)

            quality_result = self._check_quality(agent_response, metadata)

            if not quality_result["is_valid"]:
                agent_response = quality_result["fallback"]
                metadata.update(quality_result.get("metadata", {}))

            formatted_response = self._format_response(agent_response, metadata)

            return {
                "response": formatted_response,
                "metadata": metadata,
                "presentation_quality": self._assess_presentation_quality(formatted_response, metadata)
            }

        except Exception as e:
            logger.error(f"Presenter processing error: {e}")
            return {
                "response": agent_response,
                "metadata": metadata,
                "error": str(e)
            }

    def _enrich_metadata(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        result = self.metadata_enricher.process(content, metadata)
        return result.metadata

    def _check_quality(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        result = self.quality_controller.process(content, metadata)

        return {
            "is_valid": not bool(result.metadata.get("quality_issues")),
            "fallback": result.content,
            "metadata": result.metadata
        }

    def _format_response(self, content: str, metadata: Dict[str, Any]) -> str:
        format_preference = metadata.get("format_preference", "auto")

        if format_preference == "markdown" or self._should_use_markdown(content, metadata):
            result = self.markdown_formatter.process(content, metadata)
            content = result.content

        if self._should_enhance_response(content, metadata):
            result = self.response_enhancer.process(content, metadata)
            content = result.content

        result = self.text_formatter.process(content, metadata)
        return result.content

    def _should_use_markdown(self, content: str, metadata: Dict[str, Any]) -> bool:
        if metadata.get("agent_type") == "code" and "```" in content:
            return True

        if len(content.split('\n')) > 10:
            return True

        if any(word in content for word in ["def ", "class ", "import ", "from "]):
            return True

        return False

    def _should_enhance_response(self, content: str, metadata: Dict[str, Any]) -> bool:
        confidence = metadata.get("confidence", 0.5)
        word_count = len(content.split())

        if confidence < 0.4 or confidence > 0.9:
            return True

        if word_count < 5 or word_count > 200:
            return True

        return False

    def _assess_presentation_quality(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        quality_score = 0.5
        factors = []

        word_count = metadata.get("word_count", len(content.split()))
        line_count = metadata.get("line_count", len(content.split('\n')))

        if 10 <= word_count <= 300:
            quality_score += 0.2
            factors.append("appropriate_length")

        if line_count > 1 and line_count < 50:
            quality_score += 0.1
            factors.append("good_structure")

        if metadata.get("contains_code") and metadata.get("code_blocks", 0) > 0:
            quality_score += 0.1
            factors.append("code_formatting")

        if not metadata.get("quality_issues"):
            quality_score += 0.2
            factors.append("no_quality_issues")

        if content.strip() and not content.startswith("I don't"):
            quality_score += 0.1
            factors.append("substantive_content")

        quality_score = min(1.0, quality_score)

        return {
            "score": quality_score,
            "level": self._get_quality_level(quality_score),
            "factors": factors,
            "recommendations": self._get_quality_recommendations(quality_score, metadata)
        }

    def _get_quality_level(self, score: float) -> str:
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "acceptable"
        elif score >= 0.3:
            return "poor"
        else:
            return "very_poor"

    def _get_quality_recommendations(self, score: float, metadata: Dict[str, Any]) -> list:
        recommendations = []

        if score < 0.5:
            recommendations.append("improve_content_length")

        if metadata.get("quality_issues"):
            recommendations.extend(metadata["quality_issues"])

        if metadata.get("word_count", 0) < 5:
            recommendations.append("add_more_detail")

        if metadata.get("word_count", 0) > 500:
            recommendations.append("consider_summarizing")

        if metadata.get("agent_type") == "code" and not metadata.get("contains_code"):
            recommendations.append("add_code_examples")

        return recommendations
