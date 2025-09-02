import os
from enum import Enum
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class ComplexityLevel(Enum):
    SIMPLE = "gpt-5-nano"       # Factual queries, definitions, simple lookups
    MODERATE = "gpt-5-mini"     # Multi-step reasoning, synthesis of 2-3 sources  
    COMPLEX = "gpt-5"           # Deep analysis, multiple domains, creative tasks

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
    
    # Phoenix Configuration
    phoenix_endpoint: str = Field(default="http://localhost:6006")
    phoenix_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("PHOENIX_API_KEY"))
    
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

settings = Settings()