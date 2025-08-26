import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from agents.base_agent import BaseAgent
from agents.state import AgentState
from pydantic_models import ReviewVerdict, QualityScore
from config_utils import get_quality_config


class ReviewerAgent(BaseAgent):
    def __init__(self, chatbot, config):
        super().__init__(
            chatbot=chatbot,
            name="Reviewer",
            prompt=config["agents"]["reviewer"]["prompt"]
        )
        self.config = config
        self.quality_config = get_quality_config(config)

    def process(self, state: AgentState) -> AgentState:
        return self.review_response(state)

    def review_response(self, state: AgentState) -> AgentState:
        user_query = state["user_query"]
        research_results = state.get("research_results", [])
        generated_code = state.get("generated_code", "")

        try:
            full_response = self._compile_response(research_results, generated_code)
            review_result = self._evaluate_response(user_query, full_response)

            verdict_data = self._parse_review_result(review_result)

            needs_improvement = self._needs_improvement(verdict_data, state)

            if needs_improvement:
                state["metadata"]["improvement_feedback"] = verdict_data.reason
                state["metadata"]["quality_issues"] = self._identify_quality_issues(verdict_data)

            final_response = self._create_final_response(
                state, research_results, generated_code
            )

            state["quality_score"] = verdict_data.quality.overall
            state["final_response"] = final_response
            state["feedback_loop"] = needs_improvement
            state["metadata"]["review_verdict"] = verdict_data.verdict
            state["metadata"]["review_feedback"] = verdict_data.reason
            state["metadata"]["quality_breakdown"] = verdict_data.quality.model_dump()

            if needs_improvement:
                print(f"⚠️ Quality too low ({verdict_data.quality.overall:.1f}), requesting improvement")
            else:
                print(f"✅ Quality acceptable ({verdict_data.quality.overall:.1f}), approving response")

        except Exception as e:
            print(f"Review error: {e}")
            final_response = self._create_final_response(
                state, research_results, generated_code
            )
            state["quality_score"] = 3.0
            state["final_response"] = final_response
            state["feedback_loop"] = False
            state["metadata"]["review_error"] = str(e)

        return self._update_state(state)

    def _compile_response(self, research_results: list, generated_code: str) -> str:
        response_parts = []

        if research_results:
            response_parts.append("Documentation findings:")
            for i, result in enumerate(research_results[:3]):
                content = result.get('content', '')[:200] + "..."
                response_parts.append(f"{i + 1}. {content}")

        if generated_code:
            response_parts.append("\nGenerated code:")
            response_parts.append(generated_code)

        return "\n".join(response_parts)

    def _evaluate_response(self, query: str, response: str) -> str:
        evaluation_prompt = f"""
{self.prompt}

{self.config["agents"]["reviewer"]["evaluation_template"].format(query=query, response=response)}
"""

        messages = [{"role": "user", "content": evaluation_prompt}]
        return self._call_llm(messages, max_tokens=400, temperature=0.1)

    def _parse_review_result(self, review_text: str) -> ReviewVerdict:
        try:
            response_json = json.loads(review_text)
            return ReviewVerdict(**response_json)
        except Exception:
            return ReviewVerdict(
                verdict="ACCEPT",
                reason="Review parsing failed - defaulting to accept",
                quality=QualityScore(
                    completeness=3, accuracy=3, clarity=3,
                    practicality=3, code_quality=3, overall=3.0
                )
            )

    def _create_final_response(self, state: AgentState, research_results: list,
                               generated_code: str) -> str:

        if research_results:
            context = self._format_research_context(research_results)

            synthesis_prompt = self.config["agents"]["reviewer"]["synthesis_response"].format(context=context)

            try:
                synthesized_response = self._call_llm_with_context(state, synthesis_prompt, max_tokens=800, temperature=0.2)
                return synthesized_response
            except Exception:
                return self._create_basic_response(state["user_query"], research_results, generated_code)

        elif generated_code:
            return f"Here's a code solution for your question:\n\n{generated_code}"

        else:
            try:
                general_prompt = self.config["agents"]["reviewer"]["general_response"]
                simple_response = self._call_llm_with_context(state, general_prompt, max_tokens=500, temperature=0.3)
                return simple_response
            except Exception:
                return "I'm sorry, I couldn't generate a response for this query."

    def _format_research_context(self, research_results: list) -> str:
        context_parts = []

        for i, result in enumerate(research_results[:3], 1):
            content = result.get('content', '')
            source = result.get('source', f'Document {i}')

            if isinstance(content, str):
                clean_content = ' '.join(content.split())
                if len(clean_content) > 400:
                    clean_content = clean_content[:400] + "..."
                context_parts.append(f"[{source}]: {clean_content}")

        return '\n\n'.join(context_parts)

    def _identify_quality_issues(self, verdict_data: ReviewVerdict) -> list:
        issues = []

        if verdict_data.quality.completeness < 3:
            issues.append("Response lacks completeness - needs more comprehensive information")
        if verdict_data.quality.accuracy < 3:
            issues.append("Accuracy concerns - verify technical correctness")
        if verdict_data.quality.clarity < 3:
            issues.append("Clarity issues - explanation needs to be clearer")
        if verdict_data.quality.practicality < 3:
            issues.append("Practicality problems - needs more applicable examples")
        if verdict_data.quality.code_quality < 3:
            issues.append("Code quality issues - improve implementation")

        return issues

    def _needs_improvement(self, verdict_data: ReviewVerdict, state: AgentState) -> bool:
        current_iteration = state.get("iteration_count", 0)

        print(f"🔍 Reviewing quality: Overall={verdict_data.quality.overall:.1f}, Verdict={verdict_data.verdict}, Iteration={current_iteration}")

        if current_iteration >= 1:
            print("🛑 Max iterations reached, not improving")
            return False

        min_overall = self.quality_config["minimum_overall_score"]
        min_critical = self.quality_config["minimum_critical_aspects_score"]

        if verdict_data.quality.overall < min_overall:
            print(f"⚠️ Overall quality extremely low ({verdict_data.quality.overall:.1f} < {min_overall})")
            return True

        critical_aspects = [
            verdict_data.quality.completeness,
            verdict_data.quality.accuracy,
            verdict_data.quality.clarity
        ]
        if any(score < min_critical for score in critical_aspects):
            print(f"⚠️ Critical aspect extremely low: {critical_aspects}")
            return True

        return False

    def _create_basic_response(self, query: str, research_results: list, generated_code: str) -> str:
        response_parts = []

        if research_results and generated_code:
            response_parts.append("Based on the documentation, here's what I found:")
        elif research_results:
            response_parts.append("Based on the documentation:")
        elif generated_code:
            response_parts.append("Here's a code solution for your question:")

        if research_results:
            for result in research_results[:2]:
                content = result.get('content', '')
                if len(content) > 300:
                    content = content[:300] + "..."
                response_parts.append(f"\n{content}")

        if generated_code:
            response_parts.append(f"\n{generated_code}")

        return "\n".join(response_parts) if response_parts else "I couldn't find specific information for your query."

    def should_improve(self, state: AgentState) -> bool:
        if state.get("iteration_count", 0) >= 1:
            return False
        return state.get("feedback_loop", False)

    def is_final_response_ready(self, state: AgentState) -> bool:
        return not state.get("feedback_loop", False)
