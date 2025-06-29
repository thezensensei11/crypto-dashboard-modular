"""
Quick initialization script to fix DuckDB setup
Run this first to create the database with proper permissions
Place in: crypto-dashboard-modular/init_duckdb.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def initialize_duckdb():
    """Initialize DuckDB database with proper permissions"""
    print("Initializing DuckDB database...")
    
    try:
        # Import and initialize with write permissions
        from infrastructure.database.duckdb_manager import DuckDBManager
        
        # Create database with write permissions
        db_manager = DuckDBManager(db_path="crypto_data.duckdb", read_only=False)
        
        print("✅ DuckDB initialized successfully!")
        print(f"   Database location: {db_manager.db_path}")
        
        # Test connection
        with db_manager.get_connection() as conn:
            # Verify tables exist
            result = conn.execute("""
                SELECT COUNT(*) as table_count 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
            """).fetchone()
            
            print(f"   Tables created: {result[0]}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error initializing DuckDB: {e}")
        return False


if __name__ == "__main__":
    if initialize_duckdb():
        print("\n✅ DuckDB is ready to use!")
        print("\nNext steps:")
        print("1. Set environment variable: export USE_DUCKDB=true")
        print("2. Run verification: python verify_duckdb_setup.py")
        print("3. Sync some data: python infrastructure/collectors/binance_to_duckdb_sync.py --symbols BTCUSDT --intervals 1h --lookback 7")
        print("4. Run dashboard: streamlit run main.py")