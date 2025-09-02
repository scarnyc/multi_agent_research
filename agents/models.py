from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Status(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CitationStyle(Enum):
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"

class AgentMessage(BaseModel):
    sender: str
    recipient: str
    task_id: str
    payload: Dict[str, Any]
    priority: Priority = Priority.MEDIUM
    timestamp: datetime = Field(default_factory=datetime.now)

class Task(BaseModel):
    id: str
    description: str
    complexity: str
    assigned_agent: Optional[str] = None
    status: Status = Status.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
class Citation(BaseModel):
    content: str
    url: str
    title: str
    author: Optional[str] = None
    date: Optional[str] = None
    credibility_score: float = 0.0
    
class TaskResult(BaseModel):
    agent_id: str
    task_id: str
    status: Status
    result: Any
    citations: List[Citation] = []
    execution_time: float
    model_used: str
    tokens_used: Dict[str, int] = {}
    error: Optional[str] = None

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    score: float
    metadata: Dict[str, Any] = {}

class SearchResults(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float

class ExtractedContent(BaseModel):
    content: str
    source_url: str
    relevance_score: float
    key_points: List[str] = []

class CredibilityScore(BaseModel):
    score: float
    factors: Dict[str, float]
    warnings: List[str] = []

class Bibliography(BaseModel):
    citations: List[Citation]
    style: CitationStyle
    formatted_text: str