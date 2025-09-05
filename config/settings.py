import os
from enum import Enum
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class TaskType(Enum):
    DIRECT_ANSWER = "direct_answer"     # Factual questions that can be answered from training
    SEARCH_NEEDED = "search_needed"     # Questions requiring current/real-time information
    RESEARCH_REPORT = "research_report" # Deep analysis requiring comprehensive research

class ModelType(Enum):
    GPT5_NANO = "gpt-5-nano"
    GPT5_MINI = "gpt-5-mini"
    GPT5_REGULAR = "gpt-5"

class ReasoningEffort(Enum):
    MINIMAL = "minimal"     # Fastest, minimal reasoning tokens
    LOW = "low"            # Quick responses, basic reasoning
    MEDIUM = "medium"      # Default, balanced reasoning
    HIGH = "high"          # Thorough reasoning for complex tasks

class Verbosity(Enum):
    LOW = "low"            # Concise outputs
    MEDIUM = "medium"      # Balanced output length
    HIGH = "high"          # Detailed, comprehensive outputs

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    gpt5_regular_model: str = Field(default="gpt-5")
    gpt5_mini_model: str = Field(default="gpt-5-mini")
    gpt5_nano_model: str = Field(default="gpt-5-nano")
    
    # GPT-5 Specific Configuration
    default_reasoning_effort: ReasoningEffort = Field(default=ReasoningEffort.MEDIUM)
    default_verbosity: Verbosity = Field(default=Verbosity.MEDIUM)
    use_responses_api: bool = Field(default=True)  # Use new Responses API
    
    # Phoenix Direct SDK Configuration
    enable_phoenix_integration: bool = Field(default_factory=lambda: bool(os.getenv("PHOENIX_API_KEY")))  # Auto-enable if API key present
    phoenix_endpoint: str = Field(default_factory=lambda: os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006"))
    phoenix_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("PHOENIX_API_KEY"))
    phoenix_project_name: str = Field(default_factory=lambda: os.getenv("PHOENIX_PROJECT_NAME", "multi-agent-research"))
    
    # Application Settings
    max_concurrent_requests: int = Field(default=10)
    cache_ttl_seconds: int = Field(default=3600)
    request_timeout_seconds: int = Field(default=30)
    max_retries: int = Field(default=3)
    
    # Agent Settings
    max_iterations: int = Field(default=20)
    default_temperature: float = Field(default=0.7)
    default_max_tokens: int = Field(default=4096)
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra fields in .env
    }
    
    @property
    def model_mapping(self):
        """Model mapping for different use cases"""
        return {
            ModelType.GPT5_NANO: self.gpt5_nano_model,
            ModelType.GPT5_MINI: self.gpt5_mini_model,
            ModelType.GPT5_REGULAR: self.gpt5_regular_model
        }

settings = Settings()