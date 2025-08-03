"""
Tools for data analysis using pandas.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
import io
import sys
from contextlib import redirect_stdout, redirect_stderr


class PandasCodeExecutor:
    """Safe executor for pandas code with the current dataframe."""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.original_df = dataframe.copy()
    
    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute pandas code safely and return results."""
        
        # Create a safe execution environment
        safe_globals = {
            'pd': pd,
            'np': np,
            'df': self.df,
            'original_df': self.original_df,
            '__builtins__': {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                'print': print,
            }
        }
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'dataframe': None,
            'variables': {}
        }
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                exec(code, safe_globals)
                
                # Update the dataframe if it was modified
                if 'df' in safe_globals:
                    self.df = safe_globals['df']
                    result['dataframe'] = self.df.copy()
                
                # Capture any new variables created
                for key, value in safe_globals.items():
                    if key not in ['pd', 'np', 'df', 'original_df', '__builtins__']:
                        if isinstance(value, (str, int, float, bool, list, dict)):
                            result['variables'][key] = value
                        elif hasattr(value, 'describe'):  # For pandas objects
                            result['variables'][key] = str(value)
            
            result['success'] = True
            result['output'] = stdout_capture.getvalue()
            
        except Exception as e:
            result['error'] = str(e)
            result['output'] = stdout_capture.getvalue()
        
        if stderr_capture.getvalue():
            result['error'] += f"\nStderr: {stderr_capture.getvalue()}"
        
        return result


@tool
def load_csv_file(file_path: str, **kwargs) -> Dict[str, Any]:
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv (e.g., sep=',', encoding='utf-8')
    
    Returns:
        Dictionary with success status, dataframe info, and error if any
    """
    try:
        df = pd.read_csv(file_path, **kwargs)
        
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'null_counts': df.isnull().sum().to_dict(),
            'sample_data': df.head().to_dict('records')
        }
        
        return {
            'success': True,
            'dataframe': df,
            'info': info,
            'message': f"Successfully loaded CSV with shape {df.shape}"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"Failed to load CSV: {str(e)}"
        }


@tool
def execute_pandas_code(code: str, csv_file_path: str) -> Dict[str, Any]:
    """
    Execute pandas code on the current dataframe.
    
    Args:
        code: Python code to execute (should work with 'df' variable)
        csv_file_path: Path to the CSV file to load and work with
    
    Returns:
        Dictionary with execution results
    """
    if not csv_file_path:
        return {
            'success': False,
            'error': 'No CSV file specified. Please load a CSV file first.',
            'message': 'No CSV file available for analysis.'
        }
    
    try:
        # Load the dataframe
        current_dataframe = pd.read_csv(csv_file_path)
        executor = PandasCodeExecutor(current_dataframe)
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to load CSV file: {str(e)}',
            'message': f'Could not load CSV file: {csv_file_path}'
        }
    result = executor.execute_code(code)
    
    if result['success']:
        result['message'] = 'Code executed successfully'
    else:
        result['message'] = f"Code execution failed: {result['error']}"
    
    return result


@tool
def get_dataframe_info(csv_file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive information about the current dataframe.
    
    Args:
        csv_file_path: Path to the CSV file to analyze
    
    Returns:
        Dictionary with dataframe information
    """
    if not csv_file_path:
        return {'error': 'No CSV file specified'}
    
    try:
        # Load the dataframe
        dataframe = pd.read_csv(csv_file_path)
        # Basic info
        info = {
            'shape': dataframe.shape,
            'columns': dataframe.columns.tolist(),
            'dtypes': dataframe.dtypes.to_dict(),
            'memory_usage_mb': dataframe.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_counts': dataframe.isnull().sum().to_dict(),
            'null_percentages': (dataframe.isnull().sum() / len(dataframe) * 100).to_dict(),
        }
        
        # Statistical summary for numeric columns
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info['numeric_summary'] = dataframe[numeric_cols].describe().to_dict()
        
        # Sample data
        info['head'] = dataframe.head().to_dict('records')
        info['tail'] = dataframe.tail().to_dict('records')
        
        # Unique value counts for categorical columns
        categorical_cols = dataframe.select_dtypes(include=['object']).columns
        info['categorical_info'] = {}
        for col in categorical_cols:
            unique_count = dataframe[col].nunique()
            if unique_count <= 20:  # Only show value counts for columns with few unique values
                info['categorical_info'][col] = {
                    'unique_count': unique_count,
                    'value_counts': dataframe[col].value_counts().head(10).to_dict()
                }
            else:
                info['categorical_info'][col] = {
                    'unique_count': unique_count,
                    'sample_values': dataframe[col].dropna().unique()[:10].tolist()
                }
        
        return {
            'success': True,
            'info': info,
            'message': f"Dataframe info retrieved for {dataframe.shape[0]} rows and {dataframe.shape[1]} columns"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"Failed to get dataframe info: {str(e)}"
        }


@tool
def suggest_analysis_steps(csv_file_path: str, user_question: str) -> Dict[str, Any]:
    """
    Suggest analysis steps based on the dataframe and user question.
    
    Args:
        csv_file_path: Path to the CSV file to analyze
        user_question: The user's analysis question
    
    Returns:
        Dictionary with suggested analysis steps
    """
    if not csv_file_path:
        return {'error': 'No CSV file specified'}
    
    try:
        # Load the dataframe
        dataframe = pd.read_csv(csv_file_path)
    except Exception as e:
        return {'error': f'Failed to load CSV file: {str(e)}'}
    
    suggestions = []
    
    # Basic data exploration
    suggestions.extend([
        "1. Data Overview: Check dataframe shape, columns, and data types",
        f"2. Data Quality: Check for missing values in {dataframe.shape[1]} columns",
        "3. Statistical Summary: Generate descriptive statistics for numeric columns"
    ])
    
    # Column-specific suggestions
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_cols:
        suggestions.append(f"4. Numeric Analysis: Analyze {len(numeric_cols)} numeric columns: {numeric_cols[:3]}...")
    
    if categorical_cols:
        suggestions.append(f"5. Categorical Analysis: Analyze {len(categorical_cols)} categorical columns: {categorical_cols[:3]}...")
    
    # Question-specific suggestions
    question_lower = user_question.lower()
    if any(word in question_lower for word in ['correlation', 'relationship', 'relate']):
        suggestions.append("6. Correlation Analysis: Calculate correlations between numeric variables")
    
    if any(word in question_lower for word in ['trend', 'time', 'date', 'temporal']):
        suggestions.append("6. Time Series Analysis: Look for date/time columns and analyze trends")
    
    if any(word in question_lower for word in ['group', 'category', 'segment']):
        suggestions.append("6. Group Analysis: Group data by categorical variables and analyze patterns")
    
    if any(word in question_lower for word in ['outlier', 'anomaly', 'unusual']):
        suggestions.append("6. Outlier Detection: Identify outliers in numeric columns")
    
    if any(word in question_lower for word in ['distribution', 'histogram', 'spread']):
        suggestions.append("6. Distribution Analysis: Analyze data distributions and create visualizations")
    
    return {
        'success': True,
        'suggestions': suggestions,
        'message': 'Analysis steps suggested based on data and question'
    }