"""
Main LangGraph agent for data analysis.
"""

from typing import Dict, Any
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .tools import execute_pandas_code, get_dataframe_info, suggest_analysis_steps
from .config import Config


class DataAnalystAgent:
    """LangGraph-based data analyst agent using Mistral AI."""
    
    def __init__(self):
        """Initialize the agent with tools and LLM."""
        self.config = Config()
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid configuration. Please check your API keys.")
        
        # Initialize LLMs for different tasks
        self.llm = ChatMistralAI(
            api_key=self.config.MISTRAL_API_KEY,
            **self.config.get_model_config("default")
        )
        
        self.code_llm = ChatMistralAI(
            api_key=self.config.MISTRAL_API_KEY,
            **self.config.get_model_config("code")
        )
        
        # Tools available to the agent
        self.tools = [
            execute_pandas_code,
            get_dataframe_info,
            suggest_analysis_steps
        ]
        
        # Create tool node
        self.tool_node = ToolNode(self.tools)
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent."""
        return """You are an expert data analyst agent. Your goal is to help users analyze CSV data using pandas.

Key capabilities:
1. Load and inspect CSV files
2. Perform data wrangling and cleaning
3. Generate insights and analysis
4. Create visualizations when needed
5. Write and execute pandas code safely

Guidelines:
- Always start by understanding the data structure and quality
- Ask clarifying questions if the user's request is ambiguous
- Provide clear explanations for your analysis steps
- Show the code you're running for transparency
- Handle errors gracefully and suggest alternatives
- Focus on actionable insights

When working with data:
1. First load the CSV and inspect its structure
2. Check for data quality issues (missing values, duplicates, etc.)
3. Understand the user's specific question or analysis goal
4. Break down complex tasks into smaller steps
5. Execute analysis code step by step
6. Provide clear summaries and insights

Available tools:
- execute_pandas_code: Execute pandas code on the current dataframe
- get_dataframe_info: Get comprehensive information about the dataframe
- suggest_analysis_steps: Get suggested analysis steps based on data and question

Always use the tools to perform actual data operations. Never assume or make up data analysis results."""
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self.tool_node)
        
        # Define the flow
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            }
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        # Compile the graph
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Main agent reasoning node."""
        
        # This is handled below now
        
        # Get the current task or use the last message
        current_task = state.get("current_task", "")
        if not current_task and state["messages"]:
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                current_task = last_message.content
        
        # Add context about current dataframe to the system prompt
        system_context = self._create_system_prompt()
        
        if state.get("csv_loaded") and state.get("csv_file_path"):
            system_context += f"\n\nCurrent CSV file: {state['csv_file_path']}"
            if state.get("csv_info"):
                shape = state["csv_info"].get("shape", "unknown")
                system_context += f"\nDataframe shape: {shape}"
        
        if state.get("error"):
            system_context += f"\n\nPrevious error: {state['error']}"
        
        # Update the prompt template with enhanced system context
        enhanced_prompt = ChatPromptTemplate.from_messages([
            ("system", system_context),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Create the chain with enhanced prompt
        chain = enhanced_prompt | self.llm_with_tools
        
        # Use only the conversation messages
        all_messages = state["messages"]
        
        # Invoke the chain
        response = chain.invoke({
            "messages": all_messages
        })
        
        # Update state
        updated_state = {
            "messages": [response],
            "current_task": current_task,
            "error": None  # Clear any previous errors
        }
        
        return updated_state
    
    def _should_continue(self, state: AgentState) -> str: 
        """Decide whether to continue with tools or end."""
        
        last_message = state["messages"][-1]
        
        # If the last message has tool calls, continue with tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        
        # Otherwise, end the conversation
        return "end"
    
    
    def run(self, user_input: str, csv_file_path: str = None, thread_id: str = "default") -> Dict[str, Any]:
        """
        Run the agent with user input.
        
        Args:
            user_input: User's question or instruction
            csv_file_path: Optional path to CSV file to load
            thread_id: Thread ID for conversation history
        
        Returns:
            Dictionary with response and updated state
        """
        
        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "current_task": user_input,
            "csv_file_path": csv_file_path,
            "csv_info": {},
            "analysis_results": {},
            "error": None,
            "generated_code": [],
            "csv_loaded": False
        }
        
        # If CSV file is provided, modify the initial message to include CSV context
        if csv_file_path:
            enhanced_input = f"I have a CSV file at {csv_file_path}. {user_input}"
            initial_state["messages"] = [HumanMessage(content=enhanced_input)]
            initial_state["csv_loaded"] = True
            initial_state["csv_file_path"] = csv_file_path
        
        # Run the graph
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Stream the execution
            events = []
            for event in self.graph.stream(initial_state, config):
                events.append(event)
            
            # Get final state
            final_state = self.graph.get_state(config)
            
            # Extract the final response
            final_response = ""
            if final_state.values.get("messages"):
                last_message = final_state.values["messages"][-1]
                if isinstance(last_message, AIMessage):
                    final_response = last_message.content
            
            return {
                "response": final_response,
                "state": final_state.values,
                "events": events,
                "success": True
            }
            
        except Exception as e:
            return {
                "response": f"Error occurred: {str(e)}",
                "state": initial_state,
                "events": [],
                "success": False,
                "error": str(e)
            }
    
    def get_conversation_history(self, thread_id: str = "default") -> list:
        """Get conversation history for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            state = self.graph.get_state(config)
            return state.values.get("messages", [])
        except:
            return []