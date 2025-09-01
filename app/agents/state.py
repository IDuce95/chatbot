from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class AgentState(TypedDict):
    user_query: str
    conversation_history: List[Dict[str, str]]
    intent_classification: str
    target_agent: str
    research_results: List[str]
    generated_code: Optional[str]
    final_response: str
    quality_score: float
    feedback_loop: bool
    metadata: Dict[str, Any]

    current_agent: str
    agents_visited: List[str]
    iteration_count: int
