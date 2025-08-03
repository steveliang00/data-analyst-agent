"""
Main interface for running the Data Analyst Agent.
"""

import argparse
import os
import sys
from typing import Optional
from pathlib import Path

from .agent import DataAnalystAgent
from .config import Config


def print_banner():
    """Print a welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           Data Analyst Agent                                ║
║                      Powered by LangGraph & Mistral AI                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def validate_csv_file(file_path: str) -> bool:
    """Validate that the CSV file exists and is readable."""
    if not os.path.exists(file_path):
        print(f"Error: CSV file '{file_path}' does not exist.")
        return False
    
    if not file_path.lower().endswith('.csv'):
        print(f"Warning: File '{file_path}' does not have a .csv extension.")
    
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    max_size = Config.MAX_CSV_SIZE_MB
    
    if file_size_mb > max_size:
        print(f"Warning: CSV file is {file_size_mb:.1f}MB, which exceeds the recommended maximum of {max_size}MB.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    return True


def interactive_mode(agent: DataAnalystAgent, csv_file: Optional[str] = None):
    """Run the agent in interactive mode."""
    print("Interactive mode started. Type 'quit' or 'exit' to stop.")
    print("Type 'help' for available commands.")
    
    if csv_file:
        print(f"CSV file: {csv_file}")
    
    print("\n" + "="*80 + "\n")
    
    thread_id = "interactive_session"
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            if user_input.lower() == 'clear':
                # Start a new thread
                thread_id = f"interactive_session_{len(os.urandom(4).hex())}"
                print("Conversation history cleared.")
                continue
            
            if user_input.lower().startswith('load '):
                # Load a new CSV file
                new_csv_path = user_input[5:].strip()
                if validate_csv_file(new_csv_path):
                    csv_file = new_csv_path
                    print(f"CSV file path updated: {csv_file}")
                continue
            
            if not user_input:
                continue
            
            print("\nAgent: ", end="", flush=True)
            
            # Run the agent
            result = agent.run(
                user_input=user_input,
                csv_file_path=csv_file,
                thread_id=thread_id
            )
            
            if result["success"]:
                print(result["response"])
            else:
                print(f"Error: {result.get('error', 'Unknown error occurred')}")
            
            print("\n" + "-"*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'quit' to exit.\n")


def print_help():
    """Print help information."""
    help_text = """
Available commands:
- help: Show this help message
- clear: Clear conversation history and start fresh
- load <file_path>: Load a new CSV file
- quit/exit/q: Exit the program

Example questions you can ask:
- "What are the main characteristics of this dataset?"
- "Show me the distribution of values in column X"
- "Find correlations between numeric columns"
- "Clean the data by removing missing values"
- "Group the data by category and calculate averages"
- "Create a summary report of the key insights"

The agent will execute pandas code to analyze your data and provide insights.
    """
    print(help_text)


def single_query_mode(agent: DataAnalystAgent, query: str, csv_file: Optional[str] = None):
    """Run a single query and exit."""
    print(f"Query: {query}")
    if csv_file:
        print(f"CSV file: {csv_file}")
    
    print("\nProcessing...\n")
    
    result = agent.run(
        user_input=query,
        csv_file_path=csv_file,
        thread_id="single_query"
    )
    
    if result["success"]:
        print("Response:")
        print(result["response"])
    else:
        print(f"Error: {result.get('error', 'Unknown error occurred')}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Data Analyst Agent")
    parser.add_argument(
        "--csv", 
        type=str, 
        help="Path to CSV file to analyze"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="Single query to run (non-interactive mode)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Validate CSV file if provided
    if args.csv and not validate_csv_file(args.csv):
        sys.exit(1)
    
    # Initialize the agent
    try:
        print("Initializing agent...")
        agent = DataAnalystAgent()
        print("Agent initialized successfully!\n")
    except Exception as e:
        print(f"Failed to initialize agent: {str(e)}")
        print("Please check your configuration and API keys.")
        sys.exit(1)
    
    # Run in appropriate mode
    if args.query:
        single_query_mode(agent, args.query, args.csv)
    else:
        interactive_mode(agent, args.csv)


if __name__ == "__main__":
    main()