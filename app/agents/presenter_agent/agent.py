import logging
from typing import Dict, Any, Optional
from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PresenterAgent(BaseAgent):
    def __init__(self, chatbot_ref, config):
        prompt = config.get("agents", {}).get("presenter", {}).get("prompt", "")
        super().__init__(chatbot_ref, "Presenter", prompt)
        self.config = config
        self.agent_type = "presenter"
        self._initialize_tools()

    def _initialize_tools(self):
        from .tools import TextFormatterTool

        self.text_formatter = TextFormatterTool(self)

    def process(
        self, agent_response: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if metadata is None:
            metadata = {}

        try:
            metadata = self._enrich_metadata(agent_response, metadata)
            formatted_response = self.text_formatter.process(agent_response, metadata)
            presentation_quality = self._assess_presentation_quality(
                formatted_response, metadata
            )

            return {
                "response": formatted_response,
                "metadata": metadata,
                "presentation_quality": presentation_quality,
            }

        except Exception as e:
            logger.error(f"Presenter processing error: {e}")
            return {"response": agent_response, "metadata": metadata, "error": str(e)}

    def _enrich_metadata(
        self, content: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        enriched = metadata.copy()

        enriched["response_length"] = len(content)
        enriched["word_count"] = len(content.split())
        enriched["line_count"] = len(content.split("\n"))

        if "```" in content:
            enriched["contains_code"] = True
            enriched["code_blocks"] = content.count("```") // 2

        return enriched

    def _assess_presentation_quality(
        self, content: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        quality_score = 0.5
        factors = []

        word_count = metadata.get("word_count", 0)

        if 10 <= word_count <= 300:
            quality_score += 0.3
            factors.append("appropriate_length")

        if content.strip() and not content.startswith("I don't"):
            quality_score += 0.2
            factors.append("substantive_content")

        if metadata.get("contains_code") and metadata.get("code_blocks", 0) > 0:
            quality_score += 0.2
            factors.append("code_formatting")

        quality_score = min(1.0, quality_score)

        return {
            "score": quality_score,
            "level": self._get_quality_level(quality_score),
            "factors": factors,
        }

    def _get_quality_level(self, score: float) -> str:
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "acceptable"
        else:
            return "poor"
