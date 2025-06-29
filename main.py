"""
Updated main.py to work with new infrastructure
Place in: crypto-dashboard/main.py
"""

#!/usr/bin/env python3

import sys
import logging
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Check if we should use the new infrastructure
USE_NEW_INFRASTRUCTURE = os.getenv("USE_NEW_INFRASTRUCTURE", "true").lower() == "true"

if USE_NEW_INFRASTRUCTURE:
    # Import and patch the data collector to use new infrastructure
    from data.infrastructure_adapter import InfrastructureAdapter
    
    # Monkey patch the BinanceDataCollector
    import data.duckdb_collector
    data.duckdb_collector.BinanceDataCollector = InfrastructureAdapter

def main():
    """Launch the Streamlit dashboard"""
    from ui.app import run_app
    run_app()

if __name__ == "__main__":
    main()

