from ..base_agent import BaseAgent
from ..state import AgentState


class ResearchAgent(BaseAgent):
    def __init__(self, chatbot, config):
        super().__init__(
            chatbot=chatbot,
            name="Research",
            prompt=config["agents"]["research"]["prompt"]
        )
        self.config = config
        self.rag_manager = chatbot.rag_manager

    def process(self, state: AgentState) -> AgentState:
        return self.conduct_research(state)

    def conduct_research(self, state: AgentState) -> AgentState:
        user_query = state["user_query"]

        try:
            improvement_feedback = state.get("metadata", {}).get("improvement_feedback", "")
            quality_issues = state.get("metadata", {}).get("quality_issues", [])
            iteration_count = state.get("iteration_count", 0)

            if iteration_count > 0 and improvement_feedback:
                print(f"ResearchAgent: Retry with feedback: {improvement_feedback}")
                enhanced_query = self._enhance_query_with_feedback(user_query, improvement_feedback, quality_issues)
                refined_query = self._refine_query(enhanced_query)
            else:
                print("ResearchAgent: Refining search query...")
                refined_query = self._refine_query(user_query)
                print(f"ResearchAgent: Query refined to: '{refined_query[:60]}...'")

            if self.rag_manager:
                print("ResearchAgent: Searching documentation with RAG manager...")
                context, has_relevant_docs, num_sources, source_files = self.rag_manager.get_context_with_relevance(refined_query)

                if has_relevant_docs and context:
                    print(f"ResearchAgent: Found relevant context from {num_sources} sources")

                    research_results = [{
                        'content': context,
                        'source': 'Documentation',
                        'relevance': 0.8,
                        'rank': 1
                    }]

                    relevance_score = self._evaluate_relevance(research_results)

                    state["research_results"] = research_results
                    state["metadata"]["relevance_score"] = relevance_score
                    state["metadata"]["refined_query"] = refined_query
                    state["metadata"]["num_sources"] = num_sources
                    state["metadata"]["source_files"] = source_files
                else:
                    print("❌ ResearchAgent: No relevant documents found in vector database")
                    state["research_results"] = []
                    state["metadata"]["relevance_score"] = 0.0
                    state["metadata"]["no_relevant_docs"] = True
            else:
                print("❌ ResearchAgent: RAG manager not available")
                state["research_results"] = []
                state["metadata"]["rag_unavailable"] = True

        except Exception as e:
            print(f"❌ ResearchAgent error: {e}")
            state["research_results"] = []
            state["metadata"]["research_error"] = str(e)

        return self._update_state(state)

    def _enhance_query_with_feedback(self, original_query: str, feedback: str, issues: list) -> str:
        enhancements = []

        if "completeness" in feedback.lower() or any("completeness" in issue for issue in issues):
            enhancements.append("comprehensive detailed")
        if "accuracy" in feedback.lower() or any("accuracy" in issue for issue in issues):
            enhancements.append("accurate precise")
        if "clarity" in feedback.lower() or any("clarity" in issue for issue in issues):
            enhancements.append("clear explanatory")
        if "practical" in feedback.lower() or any("practical" in issue for issue in issues):
            enhancements.append("practical examples")
        if "code" in feedback.lower() or any("code" in issue for issue in issues):
            enhancements.append("code implementation")

        if enhancements:
            enhanced = f"{original_query} - need {' '.join(enhancements)} information"
            return enhanced
        else:
            return f"{original_query} - more detailed information needed"

    def _refine_query(self, query: str) -> str:
        refinement_prompt = self.config["agents"]["research"]["query_refinement"].format(query=query)

        try:
            messages = [{"role": "user", "content": refinement_prompt}]
            refined = self._call_llm(messages, max_tokens=100, temperature=0.1)
            return refined if refined else query
        except Exception:
            return query

    def _evaluate_relevance(self, results: list) -> float:
        if not results:
            return 0.0

        total_relevance = sum(item.get('relevance', 0.5) for item in results)
        avg_relevance = total_relevance / len(results)

        length_factor = min(len(results) / 3.0, 1.0)
        final_score = avg_relevance * length_factor
        return min(final_score, 1.0)

    def has_sufficient_results(self, state: AgentState) -> bool:
        results = state.get("research_results", [])
        relevance_score = state.get("metadata", {}).get("relevance_score", 0.0)
        return len(results) > 0 and relevance_score > 0.3
