#!/usr/bin/env python3
"""
Example usage of the Data Analyst Agent.
"""

import pandas as pd
import numpy as np
from src import DataAnalystAgent


def create_sample_data():
    """Create a sample CSV file for demonstration."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample sales data
    n_records = 1000
    
    data = {
        'date': pd.date_range('2023-01-01', periods=n_records, freq='D'),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_records),
        'product_name': [f'Product_{i}' for i in range(n_records)],
        'price': np.random.normal(50, 20, n_records).round(2),
        'quantity_sold': np.random.poisson(5, n_records),
        'customer_age': np.random.normal(35, 12, n_records).round().astype(int),
        'customer_gender': np.random.choice(['M', 'F', 'Other'], n_records),
        'sales_channel': np.random.choice(['Online', 'Store', 'Phone'], n_records),
        'discount_percentage': np.random.uniform(0, 30, n_records).round(1),
        'customer_satisfaction': np.random.uniform(1, 5, n_records).round(1)
    }
    
    df = pd.DataFrame(data)
    
    # Add some calculated fields
    df['total_revenue'] = df['price'] * df['quantity_sold'] * (1 - df['discount_percentage'] / 100)
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_name()
    
    # Introduce some missing values for realistic data
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'customer_satisfaction'] = np.nan
    
    # Ensure non-negative prices
    df['price'] = df['price'].clip(lower=1)
    
    return df


def run_example():
    """Run an example analysis session."""
    
    print("Creating sample data...")
    sample_df = create_sample_data()
    
    # Save to CSV
    csv_path = "data/sample_sales_data.csv"
    sample_df.to_csv(csv_path, index=False)
    print(f"Sample data saved to {csv_path}")
    
    print("\nInitializing Data Analyst Agent...")
    
    try:
        agent = DataAnalystAgent()
        print("Agent initialized successfully!")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Please make sure you have set your MISTRAL_API_KEY in a .env file")
        return
    
    # Example queries to demonstrate capabilities
    example_queries = [
        "Load the sales data and give me a comprehensive overview of the dataset",
        "What are the top-selling product categories by total revenue?",
        "Analyze the relationship between customer age and purchase behavior",
        "Find any interesting patterns in sales by day of the week",
        "Create a summary of key insights from this sales data"
    ]
    
    print("\n" + "="*80)
    print("Running Example Analysis Queries")
    print("="*80 + "\n")
    
    for i, query in enumerate(example_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 60)
        
        result = agent.run(
            user_input=query,
            csv_file_path=csv_path,
            thread_id="example_session"
        )
        
        if result["success"]:
            print("Response:")
            print(result["response"])
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "="*80 + "\n")
        
        # Add a pause between queries for readability
        input("Press Enter to continue to the next query...")


if __name__ == "__main__":
    run_example()