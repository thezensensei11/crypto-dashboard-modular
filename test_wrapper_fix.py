#!/usr/bin/env python3
"""Test if the fixes worked"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from infrastructure.database.duckdb_manager import get_db_manager
    from data.duckdb_collector import BinanceDataCollector
    
    print("Testing fixes...")
    
    # Test 1: ConnectionWrapper
    db_manager = get_db_manager()
    with db_manager.get_connection() as conn:
        # Test that register works
        import pandas as pd
        test_df = pd.DataFrame({'test': [1, 2, 3]})
        conn.register('test_table', test_df)
        result = conn.execute("SELECT * FROM test_table").fetchall()
        conn.unregister('test_table')
        print(f"✓ ConnectionWrapper register/unregister works: {len(result)} rows")
    
    # Test 2: Data collection
    collector = BinanceDataCollector()
    data = collector.get_price_data('BTCUSDT', '1h', lookback_days=1, force_cache=False)
    print(f"✓ Data collection works: {len(data)} rows fetched")
    
    # Check if data was saved
    stats = collector.get_cache_stats()
    print(f"✓ Database stats: {stats['total_rows']} total rows")
    
    print("\n✅ All fixes working correctly!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
