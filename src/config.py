"""
Configuration settings for the Data Analyst Agent.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the agent."""
    
    # Mistral AI API Configuration
    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")
    
    # Alternative models for different tasks
    MISTRAL_FAST_MODEL: str = os.getenv("MISTRAL_FAST_MODEL", "mistral-small-latest")
    MISTRAL_REASONING_MODEL: str = os.getenv("MISTRAL_REASONING_MODEL", "magistral-small-latest")
    
    # Agent Configuration
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))
    VERBOSE: bool = os.getenv("VERBOSE", "false").lower() == "true"
    
    # Data Configuration
    MAX_CSV_SIZE_MB: int = int(os.getenv("MAX_CSV_SIZE_MB", "100"))
    DEFAULT_SAMPLE_SIZE: int = int(os.getenv("DEFAULT_SAMPLE_SIZE", "1000"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.MISTRAL_API_KEY:
            print("Warning: MISTRAL_API_KEY not found in environment variables.")
            print("Please set your Mistral API key in a .env file or environment variable.")
            return False
        return True
    
    @classmethod
    def get_model_config(cls, task_type: str = "default") -> dict:
        """Get model configuration based on task type."""
        model_configs = {
            "default": {
                "model": cls.MISTRAL_MODEL,
                "temperature": 0.5,
                "max_tokens": 2000,
            },
            "fast": {
                "model": cls.MISTRAL_FAST_MODEL,
                "temperature": 0.5,
                "max_tokens": 1000,
            },
            "reasoning": {
                "model": cls.MISTRAL_REASONING_MODEL,
                "temperature": 0.5,
                "max_tokens": 3000,
            },
            "code": {
                "model": cls.MISTRAL_MODEL,
                "temperature": 0.5,
                "max_tokens": 1500,
            }
        }
        
        return model_configs.get(task_type, model_configs["default"])