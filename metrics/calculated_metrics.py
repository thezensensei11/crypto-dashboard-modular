"""
Calculated metrics module
Handles formula-based columns
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any
import logging

from crypto_dashboard_modular.utils.validators import validate_formula, extract_dependencies

logger = logging.getLogger(__name__)

class CalculatedMetrics:
    """Handle calculated/formula-based metrics"""
    
    @staticmethod
    def calculate_value(formula: str, row_data: Dict[str, float]) -> float:
        """
        Calculate a formula value for a single row
        
        Args:
            formula: Formula string with column references in brackets
            row_data: Dictionary of column values
            
        Returns:
            Calculated value
        """
        # Replace column references with actual values
        calc_formula = formula
        column_pattern = r'\[([^\]]+)\]'
        
        for match in re.finditer(column_pattern, formula):
            col_name = match.group(1)
            value = row_data.get(col_name, np.nan)
            if pd.isna(value):
                return np.nan
            calc_formula = calc_formula.replace(f'[{col_name}]', str(value))
        
        try:
            result = eval(calc_formula)
            return float(result)
        except Exception as e:
            logger.error(f"Error evaluating formula '{formula}': {e}")
            return np.nan
    
    @staticmethod
    def calculate_batch(
        dataframe: pd.DataFrame,
        calculated_columns: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Calculate all formula columns for a dataframe
        
        Args:
            dataframe: DataFrame with existing columns
            calculated_columns: List of calculated column configurations
            
        Returns:
            DataFrame with calculated columns added
        """
        df = dataframe.copy()
        
        # Sort columns by dependencies to ensure correct calculation order
        sorted_columns = CalculatedMetrics._sort_by_dependencies(calculated_columns)
        
        for col_config in sorted_columns:
            col_name = col_config['name']
            formula = col_config['formula']
            
            # Calculate values for each row
            values = []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                value = CalculatedMetrics.calculate_value(formula, row_dict)
                values.append(value)
            
            df[col_name] = values
            logger.info(f"Calculated column '{col_name}' with formula: {formula}")
        
        return df
    
    @staticmethod
    def _sort_by_dependencies(columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort columns by their dependencies to ensure correct calculation order"""
        # Build dependency graph
        deps_graph = {}
        for col in columns:
            deps_graph[col['name']] = col.get('dependencies', [])
        
        # Topological sort
        sorted_names = []
        visited = set()
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            
            # Visit dependencies first
            for dep in deps_graph.get(name, []):
                if dep in deps_graph:  # Only if it's also a calculated column
                    visit(dep)
            
            sorted_names.append(name)
        
        # Visit all nodes
        for col in columns:
            visit(col['name'])
        
        # Return columns in sorted order
        name_to_col = {col['name']: col for col in columns}
        return [name_to_col[name] for name in sorted_names if name in name_to_col]
    
    @staticmethod
    def validate_formulas(
        calculated_columns: List[Dict[str, Any]],
        available_columns: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Validate all calculated column formulas
        
        Args:
            calculated_columns: List of calculated column configurations
            available_columns: List of available column names
            
        Returns:
            List of validation results
        """
        results = []
        
        for col in calculated_columns:
            is_valid, error_msg = validate_formula(col['formula'], available_columns)
            results.append({
                'column': col['name'],
                'valid': is_valid,
                'error': error_msg if not is_valid else None,
                'dependencies': extract_dependencies(col['formula'])
            })
        
        return results
