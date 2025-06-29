#!/bin/bash
# Fix DuckDB Lock Issue
# Save as: fix_duckdb_lock.sh and run with: bash fix_duckdb_lock.sh

echo "=== Fixing DuckDB Lock Issue ==="
echo

# Colors
GREEN='\033[92m'
YELLOW='\033[93m'
RED='\033[91m'
RESET='\033[0m'

# 1. Stop all Python processes using the database
echo -e "${YELLOW}Stopping all processes using crypto_data.duckdb...${RESET}"

# Find PIDs using the database file
PIDS=$(lsof 2>/dev/null | grep crypto_data.duckdb | awk '{print $2}' | sort -u)

if [ -n "$PIDS" ]; then
    echo "Found processes: $PIDS"
    for PID in $PIDS; do
        echo "Killing PID: $PID"
        kill -9 $PID 2>/dev/null || sudo kill -9 $PID 2>/dev/null
    done
    echo -e "${GREEN}✓ Processes killed${RESET}"
else
    echo "No processes found using the database"
fi

# 2. Alternative: Kill all crypto-dashboard processes
echo
echo -e "${YELLOW}Killing any crypto-dashboard processes...${RESET}"
pkill -f "crypto-dashboard-modular" 2>/dev/null
pkill -f "run_processor" 2>/dev/null
pkill -f "run_collector" 2>/dev/null

# 3. Remove lock files if they exist
echo
echo -e "${YELLOW}Removing any lock files...${RESET}"
rm -f crypto_data.duckdb.wal 2>/dev/null
rm -f crypto_data.duckdb.lock 2>/dev/null

# 4. Wait a moment
sleep 2

# 5. Verify database is accessible
echo
echo -e "${YELLOW}Testing database access...${RESET}"
python3 -c "
import duckdb
try:
    conn = duckdb.connect('crypto_data.duckdb', read_only=True)
    tables = conn.execute('SHOW TABLES').fetchall()
    print('✓ Database is accessible. Tables:', [t[0] for t in tables])
    conn.close()
except Exception as e:
    print('✗ Database still locked:', e)
" 2>&1

echo
echo -e "${GREEN}=== Lock should be cleared ===${RESET}"
echo
echo "You can now run:"
echo "  python3 run_processor_fixed.py"
echo
echo "Or to use the original processor:"
echo "  python3 scripts/run_processor.py"