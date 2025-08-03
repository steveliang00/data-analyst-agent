# Data Analyst Agent

A LangGraph-based data analyst agent that uses Mistral AI models to analyze CSV data with pandas. The agent can take human instructions and perform sophisticated data wrangling, analysis, and insights generation.

## Features

- 🤖 **AI-Powered Analysis**: Uses Mistral AI models for intelligent data analysis
- 📊 **Pandas Integration**: Full pandas capabilities for data manipulation
- 🔄 **Interactive Conversations**: Maintains conversation history for complex analysis workflows
- 🛠️ **Safe Code Execution**: Secure execution environment for pandas operations
- 📈 **Comprehensive Analysis**: Automatic data profiling, quality checks, and insights
- 🎯 **Task-Specific Models**: Different Mistral models optimized for different tasks

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd data-analyst-agent
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate data-analyst-agent
```

3. Set up your Mistral AI API key:
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Mistral API key
# MISTRAL_API_KEY=your_api_key_here
```

## Quick Start

### Interactive Mode

Run the agent interactively to have a conversation about your data:

```bash
python -m src.main --csv data/your_file.csv
```

### Single Query Mode

Run a single analysis query:

```bash
python -m src.main --csv data/your_file.csv --query "What are the main patterns in this dataset?"
```

### Programmatic Usage

```python
from src import DataAnalystAgent

# Initialize the agent
agent = DataAnalystAgent()

# Run analysis
result = agent.run(
    user_input="Analyze the correlation between price and sales",
    csv_file_path="data/sales_data.csv"
)

print(result["response"])
```

## Example Usage

### Data Exploration
```
You: Load the sales data and give me an overview
Agent: I'll load the CSV file and provide a comprehensive overview...

[Loads data and provides summary statistics, data types, missing values, etc.]
```

### Data Analysis
```
You: What's the correlation between price and quantity sold?
Agent: Let me calculate the correlation and analyze the relationship...

[Executes pandas code and provides insights about the correlation]
```

### Data Cleaning
```
You: Clean this dataset by removing duplicates and handling missing values
Agent: I'll clean the data step by step...

[Shows the cleaning process and results]
```

## Available Commands (Interactive Mode)

- `help` - Show available commands
- `clear` - Clear conversation history
- `load <file_path>` - Load a new CSV file
- `quit` / `exit` / `q` - Exit the program

## Configuration

The agent can be configured through environment variables or the `.env` file:

```bash
# Required
MISTRAL_API_KEY=your_mistral_api_key

# Optional model selection
MISTRAL_MODEL=mistral-large-latest
MISTRAL_FAST_MODEL=mistral-small-latest
MISTRAL_REASONING_MODEL=mistral-large-latest

# Optional agent settings
MAX_ITERATIONS=10
VERBOSE=false
MAX_CSV_SIZE_MB=100
DEFAULT_SAMPLE_SIZE=1000
```

## Agent Capabilities

The agent can perform various data analysis tasks:

### Data Loading & Inspection
- Load CSV files with various encodings and separators
- Automatic data type detection
- Memory usage analysis
- Sample data preview

### Data Quality Assessment
- Missing value analysis
- Duplicate detection
- Data type validation
- Statistical summaries

### Data Analysis
- Descriptive statistics
- Correlation analysis
- Distribution analysis
- Outlier detection
- Group-by operations

### Data Visualization Suggestions
- Recommend appropriate chart types
- Generate matplotlib/seaborn code
- Create summary visualizations

### Data Transformation
- Data cleaning operations
- Feature engineering
- Data aggregation
- Filtering and sorting

## Architecture

The agent is built using:

- **LangGraph**: For workflow orchestration and state management
- **Mistral AI**: For natural language understanding and code generation
- **LangChain**: For tool integration and prompt management
- **Pandas**: For data manipulation and analysis

### Component Overview

- `src/agent.py` - Main agent class with LangGraph workflow
- `src/tools.py` - Pandas tools for data operations
- `src/state.py` - Agent state management
- `src/config.py` - Configuration management
- `src/main.py` - CLI interface

## Error Handling

The agent includes robust error handling:

- Safe code execution environment
- Graceful error recovery
- Clear error messages and suggestions
- Automatic retry mechanisms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

No License

## Support

For issues and questions:
1. Check the existing issues in the repository
2. Create a new issue with detailed information
3. Include sample data and error messages when applicable