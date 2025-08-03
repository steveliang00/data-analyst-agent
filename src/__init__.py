"""
Data Analyst Agent - A LangGraph-based data analysis agent using Mistral AI.
"""

from .agent import DataAnalystAgent
from .config import Config
from .state import AgentState

__version__ = "1.0.0"
__all__ = ["DataAnalystAgent", "Config", "AgentState"]