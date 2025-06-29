#!/usr/bin/env python3
"""
Test script to verify concurrent DuckDB access works
Save as: test_concurrent_access.py
"""

import duckdb
import time
import threading
import multiprocessing
from datetime import datetime


def test_read_only_access(db_path="crypto_data.duckdb"):
    """Test read-only access to database"""
    try:
        # Use read_only=True to allow concurrent access
        conn = duckdb.connect(db_path, read_only=True)
        tables = conn.execute("SHOW TABLES").fetchall()
        count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        conn.close()
        
        print(f"✓ Read-only access successful. Tables: {len(tables)}, Rows: {count}")
        return True
    except Exception as e:
        print(f"✗ Read-only access failed: {e}")
        return False


def test_write_with_release(db_path="crypto_data.duckdb"):
    """Test write access with connection release"""
    try:
        # Open connection
        conn = duckdb.connect(db_path)
        
        # Do a quick write
        conn.execute("""
            INSERT INTO ohlcv (symbol, interval, timestamp, open, high, low, close, volume)
            VALUES ('TEST', '1m', CURRENT_TIMESTAMP, 100, 101, 99, 100.5, 1000)
        """)
        
        # IMPORTANT: Close connection immediately after write
        conn.close()
        
        print("✓ Write completed and connection released")
        return True
    except Exception as e:
        print(f"✗ Write failed: {e}")
        return False


def concurrent_reader(name, results, db_path="crypto_data.duckdb"):
    """Function for concurrent read testing"""
    success = 0
    failures = 0
    
    for i in range(5):
        try:
            conn = duckdb.connect(db_path, read_only=True)
            conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()
            conn.close()
            success += 1
        except Exception as e:
            failures += 1
            print(f"{name}: Read failed - {e}")
        
        time.sleep(0.1)
    
    results[name] = {'success': success, 'failures': failures}


def main():
    print("=== Testing Concurrent DuckDB Access ===\n")
    
    # Test 1: Basic read-only access
    print("1. Testing read-only access:")
    test_read_only_access()
    
    # Test 2: Write and release
    print("\n2. Testing write with immediate release:")
    test_write_with_release()
    
    # Test 3: Concurrent read-only access
    print("\n3. Testing concurrent read-only access:")
    
    # Start a writer that holds connection briefly
    def writer_process():
        for i in range(3):
            conn = duckdb.connect("crypto_data.duckdb")
            conn.execute(f"INSERT INTO ohlcv (symbol, interval, timestamp, open, high, low, close, volume) VALUES ('TEST{i}', '1m', CURRENT_TIMESTAMP, 100, 101, 99, 100.5, 1000)")
            conn.close()  # Release immediately
            print(f"   Writer: Wrote record {i} and released connection")
            time.sleep(1)
    
    # Start writer in thread
    writer = threading.Thread(target=writer_process)
    writer.start()
    
    # Start multiple readers
    time.sleep(0.5)  # Let writer start
    
    results = {}
    readers = []
    for i in range(3):
        reader = threading.Thread(
            target=concurrent_reader,
            args=(f"Reader-{i}", results)
        )
        readers.append(reader)
        reader.start()
    
    # Wait for all to complete
    writer.join()
    for reader in readers:
        reader.join()
    
    # Show results
    print("\n4. Concurrent access results:")
    for name, result in results.items():
        print(f"   {name}: {result['success']} successful reads, {result['failures']} failures")
    
    # Test 4: Connection modes comparison
    print("\n5. Connection modes comparison:")
    
    # Try exclusive write (will block others)
    print("   a) Exclusive write connection (blocks others):")
    conn = duckdb.connect("crypto_data.duckdb")
    print("      - Write connection open")
    
    # Try to read while write is open
    try:
        read_conn = duckdb.connect("crypto_data.duckdb", read_only=False)
        print("      - ✗ Another connection succeeded (unexpected)")
        read_conn.close()
    except Exception as e:
        print("      - ✓ Another connection blocked (expected)")
    
    # But read-only should work
    try:
        read_conn = duckdb.connect("crypto_data.duckdb", read_only=True)
        print("      - ✓ Read-only connection succeeded")
        read_conn.close()
    except Exception as e:
        print(f"      - ✗ Read-only connection failed: {e}")
    
    conn.close()
    print("      - Write connection closed")
    
    print("\n=== Summary ===")
    print("• Use read_only=True for all read operations")
    print("• Close write connections immediately after use")
    print("• Consider batching writes to minimize lock time")
    print("• The processor should release connections between writes")


if __name__ == "__main__":
    main()