from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    response: str
    success: bool
    rag_used: Optional[bool] = False
    agents_used: Optional[List[str]] = []
    intent: Optional[str] = ""
    quality_score: Optional[float] = 0.0
    research_results: Optional[List[Dict[str, Any]]] = []
    metadata: Optional[Dict[str, Any]] = {}


class HistoryResponse(BaseModel):
    history: List[Dict[str, str]]
    count: int


class StatusResponse(BaseModel):
    status: str
    model_info: str
    agent_system_active: bool


class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]


class RouterDecision(BaseModel):
    intent: Literal["DOCUMENTATION", "CODE", "GENERAL"] = Field(
        description="Classification of user query intent"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0 and 1"
    )


class QualityScore(BaseModel):
    completeness: int = Field(ge=1, le=5, description="Completeness score 1-5")
    accuracy: int = Field(ge=1, le=5, description="Accuracy score 1-5")
    clarity: int = Field(ge=1, le=5, description="Clarity score 1-5")
    practicality: int = Field(ge=1, le=5, description="Practicality score 1-5")
    code_quality: int = Field(ge=1, le=5, description="Code quality score 1-5")
    overall: float = Field(ge=1.0, le=5.0, description="Overall score 1-5")


class ReviewVerdict(BaseModel):
    verdict: Literal["ACCEPT", "IMPROVE"] = Field(
        description="Whether to accept or improve the response"
    )
    reason: str = Field(description="Reason for the verdict")
    quality: QualityScore = Field(description="Quality breakdown")


class ResponseFormat(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None
    format_type: str = "text"
