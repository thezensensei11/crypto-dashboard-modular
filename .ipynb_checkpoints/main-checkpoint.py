#!/usr/bin/env python3
"""
Crypto Dashboard - Main Entry Point
Simple launcher that delegates to the UI module
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Launch the Streamlit dashboard"""
    from ui.app import run_app
    run_app()

if __name__ == "__main__":
    main()
