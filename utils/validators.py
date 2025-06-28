"""
Validation utilities for the dashboard
"""

import re
from typing import List, Tuple, Set

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

def validate_symbol(symbol: str) -> bool:
    """Validate a trading symbol"""
    # Basic validation - alphanumeric and ends with USDT
    pattern = re.compile(r'^[A-Z0-9]+USDT$')
    return bool(pattern.match(symbol.upper()))

def validate_interval(interval: str) -> bool:
    """Validate a time interval"""
    valid_intervals = {'1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'}
    return interval in valid_intervals

def validate_metric_name(name: str) -> Tuple[bool, str]:
    """Validate a metric name"""
    if not name:
        return False, "Name cannot be empty"
    
    if len(name) > 50:
        return False, "Name too long (max 50 characters)"
    
    # Allow alphanumeric, spaces, parentheses, and basic punctuation
    pattern = re.compile(r'^[a-zA-Z0-9\s\(\)\-\.]+$')
    if not pattern.match(name):
        return False, "Name contains invalid characters"
    
    return True, ""

def check_circular_dependencies(
    new_formula: str,
    new_column_name: str,
    existing_columns: List[dict]
) -> Tuple[bool, str]:
    """Check for circular dependencies in calculated columns"""
    # Build dependency graph
    dependencies = {new_column_name: extract_dependencies(new_formula)}
    
    for col in existing_columns:
        if col.get('type') == 'calculated':
            dependencies[col['name']] = extract_dependencies(col.get('formula', ''))
    
    # Check for cycles using DFS
    def has_cycle(node: str, visited: Set[str], rec_stack: Set[str]) -> bool:
        visited.add(node)
        rec_stack.add(node)
        
        for dep in dependencies.get(node, []):
            if dep not in visited:
                if has_cycle(dep, visited, rec_stack):
                    return True
            elif dep in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    visited = set()
    rec_stack = set()
    
    if has_cycle(new_column_name, visited, rec_stack):
        return False, "Circular dependency detected"
    
    return True, ""

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe storage"""
    # Remove/replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    # Limit length
    return sanitized[:255]
