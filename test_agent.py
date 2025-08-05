#!/usr/bin/env python3
"""
Simple test script for the Data Analyst Agent.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.config import Config
    from src.tools import get_dataframe_info
    import pandas as pd
    import numpy as np
    
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you've activated the conda environment: conda activate data-analyst-agent")
    sys.exit(1)


def test_config():
    """Test configuration setup."""
    print("\n1. Testing Configuration...")
    
    config = Config()
    
    if config.MISTRAL_API_KEY:
        print("✓ MISTRAL_API_KEY is set")
    else:
        print("⚠ MISTRAL_API_KEY is not set (you'll need this to run the agent)")
    
    print(f"✓ Default model: {config.MISTRAL_MODEL}")
    print(f"✓ Fast model: {config.MISTRAL_FAST_MODEL}")
    print(f"✓ Max iterations: {config.MAX_ITERATIONS}")


def test_tools():
    """Test pandas tools."""
    print("\n2. Testing Tools...")
    
    # Create a simple test DataFrame
    test_data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'age': [25, 30, 35, 28],
        'salary': [50000, 60000, 70000, 55000],
        'department': ['Engineering', 'Sales', 'Engineering', 'Marketing']
    }
    test_df = pd.DataFrame(test_data)
    
    # Test CSV creation and loading first
    print("\n3. Testing CSV Operations...")
    
    # Save test data to CSV
    test_csv_path = 'data/test_data.csv'
    os.makedirs('data', exist_ok=True)
    test_df.to_csv(test_csv_path, index=False)
    print(f"✓ Created test CSV: {test_csv_path}")
    
    
    # Test get_dataframe_info with CSV file
    result = get_dataframe_info.invoke({"csv_file_path": test_csv_path})
    
    if result.get('success'):
        print("✓ get_dataframe_info tool works")
        print(f"  - Shape: {result['info']['shape']}")
        print(f"  - Columns: {result['info']['columns']}")
    else:
        print(f"✗ get_dataframe_info failed: {result.get('error')}")
    
    # Clean up
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)
        print("✓ Cleaned up test file")


def test_agent_creation():
    """Test agent creation (requires API key)."""
    print("\n4. Testing Agent Creation...")
    
    try:
        from src.agent import DataAnalystAgent
        
        if not Config.MISTRAL_API_KEY:
            print("⚠ Skipping agent creation test (no API key)")
            return
        
        agent = DataAnalystAgent()
        print("✓ Agent created successfully")
        
        # Test that the agent has the required components
        assert hasattr(agent, 'llm'), "Agent should have LLM"
        assert hasattr(agent, 'tools'), "Agent should have tools"
        assert hasattr(agent, 'graph'), "Agent should have graph"
        print("✓ Agent components verified")
        
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")


def main():
    """Run all tests."""
    print("Running Data Analyst Agent Tests")
    print("=" * 50)
    
    test_config()
    test_tools()
    test_agent_creation()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("If all tests passed, you can now:")
    print("1. Set your MISTRAL_API_KEY in a .env file")
    print("2. Create sample data: python create_sample_data.py")
    print("3. Run the agent: python -m src.main --csv data/sample_sales_data.csv")


if __name__ == "__main__":
    main()