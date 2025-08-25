from pydantic import BaseModel
from typing import List, Dict, Any


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    response: str
    success: bool


class HistoryResponse(BaseModel):
    history: List[Dict[str, str]]
    count: int


class StatusResponse(BaseModel):
    status: str
    model_info: str


class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]
