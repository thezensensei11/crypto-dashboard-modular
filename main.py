#!/usr/bin/env python3
"""
Crypto Dashboard - Main Entry Point
Simple launcher that delegates to the UI module
"""


import logging




# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Launch the Streamlit dashboard"""
    from crypto_dashboard_modular.ui.app import run_app
    run_app()

if __name__ == "__main__":
    main()
