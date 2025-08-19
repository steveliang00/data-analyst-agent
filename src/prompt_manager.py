"""
Simple prompt manager for loading system prompts from YAML files.
"""
import yaml
import os
from typing import Dict, Optional
from pathlib import Path


class PromptManager:
    """Simple manager for loading system prompts from YAML files."""
    
    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt YAML files. 
                        Defaults to src/prompts/ relative to this file.
        """
        if prompts_dir is None:
            self.prompts_dir = Path(__file__).parent / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
        
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")
        
        # Cache for loaded prompts and agent metadata
        self._prompt_cache: Dict[str, str] = {}
        self._agents_metadata: Optional[Dict] = None
    
    def get_system_prompt(self, agent_type: str) -> str:
        """
        Get system prompt for a specific agent type.
        
        Args:
            agent_type: The agent type (e.g., 'coding_agent')
            
        Returns:
            The system prompt string
            
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
            KeyError: If the YAML file doesn't contain a 'system_prompt' key
        """
        # Return cached prompt if available
        if agent_type in self._prompt_cache:
            return self._prompt_cache[agent_type]
        
        # Load prompt from YAML file
        prompt_file = self.prompts_dir / f"{agent_type}.yaml"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        try:
            with open(prompt_file, 'r') as f:
                prompt_data = yaml.safe_load(f)
            
            if 'system_prompt' not in prompt_data:
                raise KeyError(f"No 'system_prompt' key found in {prompt_file}")
            
            system_prompt = prompt_data['system_prompt']
            
            # Cache the prompt
            self._prompt_cache[agent_type] = system_prompt
            
            return system_prompt
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {prompt_file}: {e}")
    
    def list_available_agents(self) -> list:
        """
        List all available agent types based on YAML files in the prompts directory.
        
        Returns:
            List of agent type names
        """
        agent_types = []
        for file_path in self.prompts_dir.glob("*.yaml"):
            # Remove the .yaml extension to get the agent type
            agent_type = file_path.stem
            # Skip the agents.yaml metadata file
            if agent_type != "agents":
                agent_types.append(agent_type)
        
        return sorted(agent_types)
    
    def _load_agents_metadata(self) -> Dict:
        """Load agent metadata from agents.yaml file."""
        if self._agents_metadata is not None:
            return self._agents_metadata
        
        agents_file = self.prompts_dir / "agents.yaml"
        if not agents_file.exists():
            # Return empty structure if no agents.yaml exists
            self._agents_metadata = {"agents": {}}
            return self._agents_metadata
        
        try:
            with open(agents_file, 'r') as f:
                self._agents_metadata = yaml.safe_load(f)
            return self._agents_metadata
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {agents_file}: {e}")
    
    def get_agent_info(self, agent_type: str) -> Dict:
        """
        Get metadata information for a specific agent type.
        
        Args:
            agent_type: The agent type (e.g., 'coding_agent')
            
        Returns:
            Dictionary containing agent metadata (name, description, prompt_file, tools)
            
        Raises:
            KeyError: If the agent type is not found in agents.yaml
        """
        metadata = self._load_agents_metadata()
        
        if agent_type not in metadata.get("agents", {}):
            # Return basic info if agent not in metadata file
            return {
                "name": agent_type.replace("_", " ").title(),
                "description": f"Agent of type {agent_type}",
                "prompt_file": f"{agent_type}.yaml",
                "tools": []
            }
        
        return metadata["agents"][agent_type]
    
    def get_agent_tools(self, agent_type: str) -> list:
        """
        Get tools for a specific agent type.
        
        Args:
            agent_type: The agent type (e.g., 'coding_agent')
            
        Returns:
            List of tool names for the agent
        """
        agent_info = self.get_agent_info(agent_type)
        return agent_info.get("tools", [])
    
    def get_all_agents_info(self) -> Dict:
        """
        Get metadata for all available agents.
        
        Returns:
            Dictionary with agent types as keys and their metadata as values
        """
        metadata = self._load_agents_metadata()
        available_agents = self.list_available_agents()
        
        result = {}
        for agent_type in available_agents:
            result[agent_type] = self.get_agent_info(agent_type)
        
        return result
    
    def clear_cache(self):
        """Clear the prompt cache and metadata cache to force reloading from files."""
        self._prompt_cache.clear()
        self._agents_metadata = None
