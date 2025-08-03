# Data Analyst Agent Demo

## Quick Start

1. **Set up your environment:**
   ```bash
   conda activate data-analyst-agent
   ```

2. **Set your Mistral API key:**
   ```bash
   echo "MISTRAL_API_KEY=your_api_key_here" > .env
   ```

3. **Test the installation:**
   ```bash
   python test_agent.py
   ```

4. **Create sample data:**
   ```bash
   python create_sample_data.py
   ```

## Usage Examples

### Single Query Mode
```bash
python -m src.main --csv data/sample_sales_data.csv --query "Give me a quick overview of this sales dataset"
```

### Interactive Mode
```bash
python -m src.main --csv data/sample_sales_data.csv
```

Then try these example queries:
- "What are the top-selling product categories?"
- "Show me the correlation between customer age and purchase amount"
- "How do sales vary by day of the week?"
- "Create a summary of customer satisfaction trends"

### Programmatic Usage
```python
from src import DataAnalystAgent

agent = DataAnalystAgent()
result = agent.run(
    user_input="Analyze sales trends by month",
    csv_file_path="data/sample_sales_data.csv"
)
print(result["response"])
```

## Sample Datasets

The agent comes with two sample datasets:

1. **Sales Data** (`data/sample_sales_data.csv`)
   - 500 records of e-commerce sales
   - Includes customer demographics, product info, pricing, and satisfaction

2. **Employee Data** (`data/sample_employee_data.csv`)
   - 200 employee records
   - Includes salary, department, performance, and experience data

## Key Features Demonstrated

✅ **CSV Loading & Analysis** - Automatic data profiling and quality assessment  
✅ **Pandas Code Execution** - Safe execution of data manipulation code  
✅ **Statistical Analysis** - Descriptive statistics and correlation analysis  
✅ **Data Visualization Suggestions** - Recommendations for charts and graphs  
✅ **Conversation Memory** - Maintains context across multiple queries  
✅ **Error Handling** - Graceful error recovery and helpful messages  

## Agent Capabilities

The agent can handle various data analysis tasks:

- **Data Exploration**: Dataset overview, column analysis, missing values
- **Statistical Analysis**: Descriptive stats, correlations, distributions
- **Data Cleaning**: Handle missing values, duplicates, data type conversion
- **Grouping & Aggregation**: Group by operations, pivot tables
- **Trend Analysis**: Time series analysis, seasonal patterns
- **Insights Generation**: Automated insights and recommendations

## Architecture

The agent uses:
- **LangGraph** for workflow orchestration
- **Mistral AI** for natural language understanding and code generation
- **Pandas** for data manipulation and analysis
- **LangChain** for tool integration and prompt management