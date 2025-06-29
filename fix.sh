#!/bin/bash
# Script to fix DuckDB lock issue

echo "=== Fixing DuckDB Lock Issue ==="
echo

# 1. Find and kill the process holding the lock
echo "1. Looking for Python processes with PID 29885..."
ps aux | grep 29885 | grep -v grep

echo
echo "2. Killing the process holding the DuckDB lock..."
kill -9 29885

# Wait a moment
sleep 1

# 2. Check if it's killed
if ps -p 29885 > /dev/null 2>&1; then
    echo "   ⚠️  Process still running, trying sudo..."
    sudo kill -9 29885
else
    echo "   ✓ Process killed successfully"
fi

echo
echo "3. Looking for any other Python processes that might be using crypto_data.duckdb..."
lsof | grep crypto_data.duckdb 2>/dev/null || echo "   No other processes found using the database"

echo
echo "4. Alternative: Kill all Python processes accessing the project..."
echo "   Run this if needed: pkill -f 'crypto-dashboard-modular'"

echo
echo "✅ DuckDB lock should be cleared. You can now run:"
echo "   python3 scripts/run_processor.py"