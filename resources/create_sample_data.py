#!/usr/bin/env python3
"""
Create sample CSV data for testing the Data Analyst Agent.
"""

import pandas as pd
import numpy as np
import os


def create_sample_sales_data():
    """Create sample sales data CSV."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate 500 records of sample sales data
    n_records = 500
    
    # Date range for 2023
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    data = {
        'order_id': [f'ORD-{str(i).zfill(6)}' for i in range(1, n_records + 1)],
        'date': np.random.choice(dates, n_records),
        'product_category': np.random.choice(
            ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Beauty'], 
            n_records, 
            p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
        ),
        'product_name': [
            np.random.choice([
                'Laptop Pro', 'Gaming Mouse', 'Wireless Headphones', 'Smart Watch', 'Tablet',
                'T-Shirt', 'Jeans', 'Sneakers', 'Dress', 'Jacket',
                'Programming Book', 'Fiction Novel', 'Cookbook', 'History Book', 'Science Magazine',
                'Garden Tools', 'Kitchen Appliance', 'Furniture', 'Decor Item', 'Storage Box',
                'Running Shoes', 'Yoga Mat', 'Dumbbells', 'Tennis Racket', 'Bicycle',
                'Skincare Set', 'Makeup Kit', 'Perfume', 'Hair Care', 'Nail Polish'
            ]) for _ in range(n_records)
        ],
        'price': np.round(np.random.lognormal(3.5, 0.8, n_records), 2),  # Log-normal distribution for realistic prices
        'quantity': np.random.randint(1, 10, n_records),
        'customer_age': np.random.normal(40, 15, n_records).astype(int).clip(18, 80),
        'customer_gender': np.random.choice(['Male', 'Female', 'Other'], n_records, p=[0.45, 0.50, 0.05]),
        'sales_channel': np.random.choice(['Online', 'In-Store', 'Mobile App'], n_records, p=[0.50, 0.35, 0.15]),
        'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash'], n_records, p=[0.40, 0.30, 0.20, 0.10]),
        'discount_applied': np.random.choice([True, False], n_records, p=[0.30, 0.70]),
        'customer_satisfaction': np.round(np.random.normal(4.0, 0.8, n_records).clip(1, 5), 1)
    }
    
    df = pd.DataFrame(data)
    
    # Add calculated fields
    df['discount_amount'] = np.where(
        df['discount_applied'], 
        np.round(df['price'] * df['quantity'] * np.random.uniform(0.05, 0.25, n_records), 2),
        0.0
    )
    df['total_amount'] = np.round(df['price'] * df['quantity'] - df['discount_amount'], 2)
    
    # Add some seasonal patterns
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Add day of week
    df['day_of_week'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.weekday >= 5
    
    # Introduce some realistic missing values (about 3% of satisfaction scores)
    missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[missing_indices, 'customer_satisfaction'] = np.nan
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def create_sample_employee_data():
    """Create sample employee data CSV."""
    
    np.random.seed(123)
    n_employees = 200
    
    data = {
        'employee_id': [f'EMP-{str(i).zfill(4)}' for i in range(1, n_employees + 1)],
        'first_name': [
            np.random.choice(['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa', 'Chris', 'Amy', 'Tom', 'Kate']) 
            for _ in range(n_employees)
        ],
        'last_name': [
            np.random.choice(['Smith', 'Johnson', 'Brown', 'Davis', 'Wilson', 'Miller', 'Moore', 'Taylor', 'Anderson', 'Thomas']) 
            for _ in range(n_employees)
        ],
        'department': np.random.choice(
            ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations'], 
            n_employees, 
            p=[0.30, 0.25, 0.15, 0.10, 0.10, 0.10]
        ),
        'position': [
            np.random.choice([
                'Senior Manager', 'Manager', 'Senior Analyst', 'Analyst', 'Associate', 'Specialist'
            ]) for _ in range(n_employees)
        ],
        'hire_date': pd.to_datetime(np.random.choice(
            pd.date_range('2018-01-01', '2023-12-31'), n_employees
        )),
        'salary': np.round(np.random.normal(75000, 25000, n_employees).clip(40000, 200000), 0),
        'age': np.random.normal(35, 10, n_employees).astype(int).clip(22, 65),
        'years_experience': np.random.normal(8, 5, n_employees).astype(int).clip(0, 40),
        'education_level': np.random.choice(
            ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], 
            n_employees, 
            p=[0.10, 0.50, 0.35, 0.05]
        ),
        'performance_rating': np.round(np.random.normal(3.5, 0.7, n_employees).clip(1, 5), 1),
        'is_remote': np.random.choice([True, False], n_employees, p=[0.40, 0.60])
    }
    
    df = pd.DataFrame(data)
    
    # Add calculated fields
    df['years_at_company'] = (pd.Timestamp.now() - df['hire_date']).dt.days / 365.25
    df['years_at_company'] = df['years_at_company'].round(1)
    
    # Introduce some correlations for more realistic data
    # Higher education generally correlates with higher salary
    education_multipliers = {
        'High School': 0.8,
        'Bachelor\'s': 1.0,
        'Master\'s': 1.3,
        'PhD': 1.6
    }
    df['salary'] = df.apply(lambda row: row['salary'] * education_multipliers[row['education_level']], axis=1)
    df['salary'] = df['salary'].round(0)
    
    return df


def main():
    """Create sample data files."""
    
    print("Creating sample data files...")
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Create sales data
    print("Creating sales data...")
    sales_df = create_sample_sales_data()
    sales_path = 'data/sample_sales_data.csv'
    sales_df.to_csv(sales_path, index=False)
    print(f"✓ Sales data saved to {sales_path} ({len(sales_df)} records)")
    
    # Create employee data
    print("Creating employee data...")
    employee_df = create_sample_employee_data()
    employee_path = 'data/sample_employee_data.csv'
    employee_df.to_csv(employee_path, index=False)
    print(f"✓ Employee data saved to {employee_path} ({len(employee_df)} records)")
    
    print("\nSample data files created successfully!")
    print("\nYou can now test the agent with:")
    print(f"  python -m src.main --csv {sales_path}")
    print(f"  python -m src.main --csv {employee_path}")


if __name__ == "__main__":
    main()