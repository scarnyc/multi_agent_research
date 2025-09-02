import os
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel
from typing import Optional

load_dotenv()

class ComplexityLevel(Enum):
    SIMPLE = "nano"      # Factual queries, definitions, simple lookups
    MODERATE = "mini"    # Multi-step reasoning, synthesis of 2-3 sources
    COMPLEX = "regular"  # Deep analysis, multiple domains, creative tasks

class Config(BaseModel):
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    gpt5_nano_model: str = "gpt-5-nano"
    gpt5_mini_model: str = "gpt-5-mini"  
    gpt5_regular_model: str = "gpt-5"
    
    default_model: str = os.getenv("DEFAULT_MODEL", "gpt-5-mini")
    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    default_max_tokens: int = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))
    
    # Agent Configuration
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "20"))
    
    @property
    def model_for_complexity(self) -> dict:
        return {
            ComplexityLevel.SIMPLE: self.gpt5_nano_model,
            ComplexityLevel.MODERATE: self.gpt5_mini_model,
            ComplexityLevel.COMPLEX: self.gpt5_regular_model
        }

config = Config()