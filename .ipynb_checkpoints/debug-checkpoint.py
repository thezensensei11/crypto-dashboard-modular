#!/usr/bin/env python3
"""
Complete system status check
Save as: check_status.py
Run with: python3 check_status.py
"""

import redis
import psutil
import os
from datetime import datetime
from pathlib import Path

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def check_processes():
    """Check running processes"""
    print(f"\n{BOLD}1. Running Processes:{RESET}")
    print("-" * 50)
    
    processes = {
        'processor': False,
        'websocket': False,
        'rest': False,
        'redis': False
    }
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            
            if 'redis-server' in cmdline:
                processes['redis'] = proc.info['pid']
            elif 'run_processor.py' in cmdline:
                processes['processor'] = proc.info['pid']
            elif 'run_collector.py websocket' in cmdline:
                processes['websocket'] = proc.info['pid']
            elif 'run_collector.py rest' in cmdline:
                processes['rest'] = proc.info['pid']
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Display status
    for name, pid in processes.items():
        if pid:
            print(f"{GREEN}✓ {name:12} running (PID: {pid}){RESET}")
        else:
            print(f"{RED}✗ {name:12} not running{RESET}")
    
    return processes


def check_redis():
    """Check Redis status and contents"""
    print(f"\n{BOLD}2. Redis Status:{RESET}")
    print("-" * 50)
    
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print(f"{GREEN}✓ Redis connected{RESET}")
        
        # Check streams
        streams = [
            "crypto:stream:price.update",
            "crypto:stream:candle.closed",
            "crypto:stream:historical.data"
        ]
        
        total_messages = 0
        for stream in streams:
            try:
                length = r.xlen(stream)
                total_messages += length
                
                # Get latest message
                latest_msg = None
                if length > 0:
                    messages = r.xrevrange(stream, count=1)
                    if messages:
                        msg_id = messages[0][0]
                        timestamp = int(msg_id.split('-')[0]) / 1000
                        age = (datetime.now().timestamp() - timestamp)
                        
                        if age < 60:
                            status = f"{GREEN}active ({int(age)}s ago){RESET}"
                        elif age < 300:
                            status = f"{YELLOW}recent ({int(age/60)}m ago){RESET}"
                        else:
                            status = f"{RED}stale ({int(age/3600)}h ago){RESET}"
                        
                        print(f"  {stream.split(':')[-1]:20} {length:6} msgs  {status}")
                else:
                    print(f"  {stream.split(':')[-1]:20} {length:6} msgs  {YELLOW}empty{RESET}")
                    
            except Exception as e:
                print(f"  {stream.split(':')[-1]:20} {RED}error: {e}{RESET}")
        
        print(f"\n  Total messages: {total_messages}")
        
    except Exception as e:
        print(f"{RED}✗ Redis error: {e}{RESET}")
        return False
    
    return True


def check_duckdb():
    """Check DuckDB status"""
    print(f"\n{BOLD}3. DuckDB Status:{RESET}")
    print("-" * 50)
    
    db_path = Path("crypto_data.duckdb")
    
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"{GREEN}✓ Database exists ({size_mb:.1f} MB){RESET}")
        
        # Check for locks
        try:
            import duckdb
            conn = duckdb.connect(str(db_path), read_only=True)
            
            # Get table info
            tables = conn.execute("SHOW TABLES").fetchall()
            print(f"  Tables: {[t[0] for t in tables]}")
            
            # Get row counts for OHLCV tables
            for table in tables:
                if table[0].startswith('ohlcv_'):
                    count = conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
                    print(f"  {table[0]}: {count:,} rows")
            
            conn.close()
            print(f"{GREEN}  No lock issues{RESET}")
            
        except Exception as e:
            print(f"{RED}  Database locked or error: {e}{RESET}")
            
            # Find process holding lock
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    for file in proc.open_files():
                        if 'crypto_data.duckdb' in file.path:
                            print(f"{YELLOW}  Locked by: {proc.info['name']} (PID: {proc.info['pid']}){RESET}")
                except:
                    pass
    else:
        print(f"{YELLOW}✗ Database does not exist yet{RESET}")


def check_last_error():
    """Check for recent errors in logs"""
    print(f"\n{BOLD}4. Recent Errors:{RESET}")
    print("-" * 50)
    
    log_file = Path("processor_debug.log")
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        errors = [l for l in lines[-100:] if 'ERROR' in l]
        if errors:
            print(f"{RED}Found {len(errors)} errors in last 100 log lines:{RESET}")
            for error in errors[-3:]:  # Show last 3 errors
                print(f"  {error.strip()}")
        else:
            print(f"{GREEN}No recent errors in logs{RESET}")
    else:
        print(f"{YELLOW}No debug log file found{RESET}")


def show_recommendations(processes):
    """Show recommendations based on status"""
    print(f"\n{BOLD}5. Recommendations:{RESET}")
    print("-" * 50)
    
    if not processes['redis']:
        print(f"{RED}1. Start Redis first:{RESET}")
        print(f"   redis-server")
    
    if not processes['processor']:
        print(f"{YELLOW}2. Start the data processor:{RESET}")
        print(f"   python3 run_processor_fixed.py  # (with format fix)")
        print(f"   # or")
        print(f"   python3 scripts/run_processor.py")
    
    if not processes['websocket']:
        print(f"{YELLOW}3. Start the WebSocket collector:{RESET}")
        print(f"   python3 scripts/run_collector.py websocket")
    
    if not processes['rest']:
        print(f"{BLUE}4. (Optional) Start the REST collector:{RESET}")
        print(f"   python3 scripts/run_collector.py rest")
    
    if all(processes.values()):
        print(f"{GREEN}✓ All services are running!{RESET}")
        print(f"\nMonitor progress with:")
        print(f"   python3 monitor_progress.py")


def main():
    """Main status check"""
    print(f"{BOLD}=== Crypto Dashboard System Status ==={RESET}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run checks
    processes = check_processes()
    check_redis()
    check_duckdb()
    check_last_error()
    show_recommendations(processes)
    
    print(f"\n{BOLD}Quick Commands:{RESET}")
    print(f"  Kill DuckDB lock:  pkill -f crypto_data.duckdb")
    print(f"  Clear Redis:       redis-cli FLUSHALL")
    print(f"  Stop all:          pkill -f crypto-dashboard-modular")


if __name__ == "__main__":
    main()