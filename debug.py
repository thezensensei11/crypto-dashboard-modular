#!/usr/bin/env python3
"""
Test script for calculated columns functionality
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def validate_formula(formula: str, available_columns: List[str]) -> Tuple[bool, str]:
    """Validate a calculated column formula"""
    if not formula:
        return False, "Formula cannot be empty"
    
    # Check for valid characters
    valid_chars = re.compile(r'^[\[\]a-zA-Z0-9_\s\+\-\*/\(\)\.]+$')
    if not valid_chars.match(formula):
        return False, "Formula contains invalid characters"
    
    # Extract column references
    column_pattern = r'\[([^\]]+)\]'
    referenced_columns = re.findall(column_pattern, formula)
    
    # Check if all referenced columns exist
    for col in referenced_columns:
        if col not in available_columns:
            return False, f"Column '{col}' not found"
    
    # Try to parse the formula
    try:
        # Replace column references with dummy values for validation
        test_formula = formula
        for col in referenced_columns:
            test_formula = test_formula.replace(f'[{col}]', '1.0')
        
        # Try to evaluate
        eval(test_formula)
        return True, ""
    except Exception as e:
        return False, f"Invalid expression: {str(e)}"

def extract_dependencies(formula: str) -> List[str]:
    """Extract column dependencies from a formula"""
    column_pattern = r'\[([^\]]+)\]'
    return list(set(re.findall(column_pattern, formula)))

def calculate_column_value(formula: str, row_data: Dict[str, float]) -> float:
    """Calculate a formula value for a single row"""
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
    except:
        return np.nan

def test_calculated_columns():
    """Test the calculated columns functionality"""
    print("=== TESTING CALCULATED COLUMNS ===\n")
    
    # Test data
    available_columns = ['beta', 'upside_beta', 'downside_beta', 'volatility', 'correlation']
    
    # Test formulas
    test_formulas = [
        ("[upside_beta] / [downside_beta]", "Beta ratio"),
        ("([beta] + [correlation]) / 2", "Average metric"),
        ("[volatility] * 100", "Volatility percentage"),
        ("[upside_beta] - [downside_beta]", "Beta spread"),
        ("1 / [beta]", "Inverse beta"),
        ("[beta] * [correlation] * [volatility]", "Composite score"),
        ("([upside_beta] > [downside_beta]) * 1", "Invalid - contains >"),
        ("[missing_column] + 1", "Invalid - missing column"),
        ("", "Invalid - empty formula"),
    ]
    
    print("1. Testing formula validation:")
    print("-" * 50)
    
    for formula, description in test_formulas:
        is_valid, error_msg = validate_formula(formula, available_columns)
        status = "✅ Valid" if is_valid else f"❌ Invalid: {error_msg}"
        print(f"{description:30} | Formula: {formula:40} | {status}")
    
    print("\n2. Testing dependency extraction:")
    print("-" * 50)
    
    valid_formulas = [f for f, _ in test_formulas[:6]]
    for formula in valid_formulas:
        deps = extract_dependencies(formula)
        print(f"Formula: {formula:50} | Dependencies: {deps}")
    
    print("\n3. Testing calculations:")
    print("-" * 50)
    
    # Sample data rows
    test_data = [
        {'beta': 1.2, 'upside_beta': 1.5, 'downside_beta': 0.8, 'volatility': 0.25, 'correlation': 0.7},
        {'beta': 0.9, 'upside_beta': 1.0, 'downside_beta': 0.8, 'volatility': 0.15, 'correlation': 0.5},
        {'beta': 1.5, 'upside_beta': 1.8, 'downside_beta': 1.2, 'volatility': 0.35, 'correlation': 0.9},
        {'beta': np.nan, 'upside_beta': 1.3, 'downside_beta': 1.1, 'volatility': 0.20, 'correlation': 0.6},
    ]
    
    # Calculate for each valid formula
    for formula in valid_formulas[:5]:  # First 5 valid formulas
        print(f"\nFormula: {formula}")
        for i, row in enumerate(test_data):
            result = calculate_column_value(formula, row)
            print(f"  Row {i+1}: {result:.4f}" if not pd.isna(result) else f"  Row {i+1}: NaN")
    
    print("\n4. Testing complex scenarios:")
    print("-" * 50)
    
    # Test with missing values
    row_with_missing = {'beta': 1.2, 'upside_beta': np.nan, 'downside_beta': 0.8}
    formula = "[upside_beta] / [downside_beta]"
    result = calculate_column_value(formula, row_with_missing)
    print(f"Formula with NaN input: {formula} = {result}")
    
    # Test division by zero
    row_with_zero = {'beta': 0, 'upside_beta': 1.5, 'downside_beta': 0.8}
    formula = "1 / [beta]"
    result = calculate_column_value(formula, row_with_zero)
    print(f"Division by zero: {formula} = {result}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_calculated_columns()