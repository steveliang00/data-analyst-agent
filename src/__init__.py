"""
Data Analyst Agent - A LangGraph-based data analysis agent using Mistral AI.
"""

from .coding_agent import DataAnalystAgent
from .state import AgentState

__version__ = "1.0.0"
__all__ = ["DataAnalystAgent", "Config", "AgentState"]