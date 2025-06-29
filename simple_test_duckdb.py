"""
Fixed simple test script for DuckDB implementation
Place in: crypto-dashboard-modular/fixed_simple_test_duckdb.py
"""

import os
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Set to use DuckDB
os.environ['USE_DUCKDB'] = 'true'

def simple_test():
    print("=" * 60)
    print("Simple DuckDB Test")
    print("=" * 60)
    
    # Step 1: Create __init__ files
    print("\n1. Creating __init__.py files...")
    try:
        from create_init_files import create_init_files
        create_init_files()
    except Exception as e:
        print(f"   Warning: Could not create init files: {e}")
    
    # Step 2: Initialize DuckDB
    print("\n2. Initializing DuckDB...")
    try:
        from infrastructure.database.duckdb_manager import DuckDBManager
        db = DuckDBManager(read_only=False)
        print("✅ DuckDB initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize DuckDB: {e}")
        return
    
    # Step 3: Test data collection
    print("\n3. Testing data collection...")
    try:
        from data.duckdb_collector import BinanceDataCollector
        
        collector = BinanceDataCollector()
        print("✅ Collector created")
        
        # Get the internal collector for stats
        internal_collector = collector._sync_collector
        
        # Test fetching a small amount of data (force_cache=False to allow API calls)
        print("\n4. Fetching BTCUSDT data (this will make API calls if no data in DB)...")
        data = collector.get_price_data('BTCUSDT', '1h', lookback_days=1, force_cache=False)
        
        if not data.empty:
            print(f"✅ Successfully fetched {len(data)} candles")
            print(f"   Latest: {data['timestamp'].max()} - Close: ${data.iloc[-1]['close']:,.2f}")
            print(f"   API calls made: {internal_collector.api_call_count}")
        else:
            print("❌ No data fetched - trying another symbol...")
            # Try ETHUSDT as backup
            data = collector.get_price_data('ETHUSDT', '1h', lookback_days=1, force_cache=False)
            if not data.empty:
                print(f"✅ Successfully fetched {len(data)} candles for ETHUSDT")
            else:
                print("❌ Still no data - Binance API might be down or network issue")
        
        # Test cache hit
        print("\n5. Testing cache (should not make API calls)...")
        
        # Reset counters on internal collector
        internal_collector.reset_counters()
        
        # Fetch same data again
        data2 = collector.get_price_data('BTCUSDT', '1h', lookback_days=1, force_cache=True)
        
        if not data2.empty:
            print(f"✅ Retrieved {len(data2)} candles from cache")
        
        # Get stats from internal collector
        print(f"✅ Cache test: API calls: {internal_collector.api_call_count}, Cache hits: {internal_collector.cache_hit_count}")
        
        # Get overall stats
        stats = collector.get_cache_stats()
        print(f"\n6. Database stats:")
        print(f"   Total symbols: {stats.get('total_symbols', 0)}")
        print(f"   Total rows: {stats.get('total_rows', 0)}")
        print(f"   Database size: {stats.get('total_size_mb', 0):.2f} MB")
        
    except Exception as e:
        print(f"❌ Error testing collector: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✅ Basic test completed!")
    print("\nNext steps:")
    print("1. If no data was fetched, you can manually sync:")
    print("   python infrastructure/collectors/binance_to_duckdb_sync.py --symbols BTCUSDT --intervals 1h --lookback 1")
    print("\n2. Run the dashboard:")
    print("   export USE_DUCKDB=true")
    print("   streamlit run main.py")
    print("=" * 60)


if __name__ == "__main__":
    simple_test()