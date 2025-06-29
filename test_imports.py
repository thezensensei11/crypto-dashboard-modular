#!/usr/bin/env python3
"""
Test script to verify all imports work correctly after package installation
Run this after installing with: pip install -e .
"""

import sys

def test_imports():
    """Test that all major imports work correctly"""
    print("Testing crypto-dashboard-modular imports...")
    print("=" * 60)
    
    tests = []
    
    # Test 1: Package version
    try:
        import crypto_dashboard_modular
        version = crypto_dashboard_modular.__version__
        tests.append(("Package version", f"✓ {version}"))
    except Exception as e:
        tests.append(("Package version", f"✗ {str(e)}"))
    
    # Test 2: Config module
    try:
        from crypto_dashboard_modular.config import get_settings, INTERVALS
        tests.append(("Config module", f"✓ Loaded {len(INTERVALS)} intervals"))
    except Exception as e:
        tests.append(("Config module", f"✗ {str(e)}"))
    
    # Test 3: Data module
    try:
        from crypto_dashboard_modular.data import BinanceDataCollector, SmartDataManager
        tests.append(("Data module", "✓ BinanceDataCollector, SmartDataManager"))
    except Exception as e:
        tests.append(("Data module", f"✗ {str(e)}"))
    
    # Test 4: Metrics module
    try:
        from metrics.unified_engine import MetricsEngine, PriceMetrics
        tests.append(("Metrics module", "✓ MetricsEngine, PriceMetrics"))
    except Exception as e:
        tests.append(("Metrics module", f"✗ {str(e)}"))
    
    # Test 5: UI module
    try:
        from crypto_dashboard_modular.ui import DashboardApp
        tests.append(("UI module", "✓ DashboardApp"))
    except Exception as e:
        tests.append(("UI module", f"✗ {str(e)}"))
    
    # Test 6: Utils module
    try:
        from crypto_dashboard_modular.utils import validate_symbol, format_price
        tests.append(("Utils module", "✓ validate_symbol, format_price"))
    except Exception as e:
        tests.append(("Utils module", f"✗ {str(e)}"))
    
    # Test 7: Legacy imports (should show deprecation warnings)
    print("\nTesting legacy imports (should show deprecation warnings):")
    print("-" * 60)
    
    try:
        import binance_data_collector
        tests.append(("Legacy binance_data_collector", "✓ (with deprecation warning)"))
    except Exception as e:
        tests.append(("Legacy binance_data_collector", f"✗ {str(e)}"))
    
    try:
        import smart_data_manager
        tests.append(("Legacy smart_data_manager", "✓ (with deprecation warning)"))
    except Exception as e:
        tests.append(("Legacy smart_data_manager", f"✗ {str(e)}"))
    
    # Print results
    print("\nImport Test Results:")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in tests:
        print(f"{test_name:<30} {result}")
        if "✓" in result:
            passed += 1
        else:
            failed += 1
    
    print("=" * 60)
    print(f"Total: {len(tests)} tests, {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)