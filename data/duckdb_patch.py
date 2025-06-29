"""
Simple patch for DuckDB column mismatch
Add this to the top of your duckdb_collector.py after the imports
"""

# Monkey-patch the insert method to handle column mismatch
_original_insert = None

def patch_duckdb_manager():
    """Patch DuckDBManager to handle column mismatch"""
    global _original_insert
    
    try:
        from infrastructure.database.duckdb_manager import get_db_manager
        db_manager = get_db_manager()
        
        if hasattr(db_manager, 'insert_ohlcv_batch'):
            _original_insert = db_manager.insert_ohlcv_batch
            
            def patched_insert(self, data, symbol, interval):
                # Add created_at if missing
                if 'created_at' not in data.columns:
                    data = data.copy()
                    data['created_at'] = pd.Timestamp.now(tz='UTC')
                
                # Ensure we have exactly the columns expected
                expected_cols = ['symbol', 'interval', 'timestamp', 'open', 'high', 
                               'low', 'close', 'volume', 'quote_volume', 'trades', 
                               'taker_buy_base', 'taker_buy_quote', 'created_at']
                
                # Add symbol and interval
                data['symbol'] = symbol
                data['interval'] = interval
                
                # Reorder and select columns
                available_cols = [col for col in expected_cols if col in data.columns]
                data = data[available_cols]
                
                return _original_insert(data, symbol, interval)
            
            # Replace method
            db_manager.__class__.insert_ohlcv_batch = patched_insert
            print("✓ Patched DuckDBManager.insert_ohlcv_batch")
            
    except Exception as e:
        print(f"Warning: Could not patch DuckDBManager: {e}")

# Apply patch when module loads
patch_duckdb_manager()
