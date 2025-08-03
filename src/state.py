"""
State management for the Data Analyst Agent.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State for the data analyst agent."""
    
    # Message history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # CSV file path (instead of storing the dataframe directly)
    csv_file_path: Optional[str]
    
    # CSV metadata
    csv_info: Dict[str, Any]
    
    # Analysis results
    analysis_results: Dict[str, Any]
    
    # Current task/instruction
    current_task: str
    
    # Error state
    error: Optional[str]
    
    # Generated code for transparency
    generated_code: List[str]
    
    # Flag to indicate if CSV is loaded
    csv_loaded: bool