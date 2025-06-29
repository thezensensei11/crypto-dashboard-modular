#!/bin/bash
# Setup script for new data infrastructure
# Run this from your project root: crypto-dashboard-modular/

echo "Setting up new data infrastructure..."
echo "===================================="

# 1. Create directory structure
echo "1. Creating directory structure..."
mkdir -p infrastructure/database
mkdir -p infrastructure/message_bus
mkdir -p infrastructure/collectors
mkdir -p infrastructure/scheduler

# 2. Create __init__.py files
echo "2. Creating __init__.py files..."
touch infrastructure/__init__.py
touch infrastructure/database/__init__.py
touch infrastructure/message_bus/__init__.py
touch infrastructure/collectors/__init__.py
touch infrastructure/scheduler/__init__.py

# 3. Check if DuckDB is installed
echo "3. Checking dependencies..."
python -c "import duckdb" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing DuckDB..."
    pip install duckdb>=0.9.0
fi

# 4. Check if Redis is available
redis-cli ping 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: Redis is not running. Some features will not work."
    echo "Install Redis with: sudo apt install redis-server (Ubuntu) or brew install redis (macOS)"
fi

# 5. Create .env.example file
echo "4. Creating .env.example..."
cat > .env.example << EOF
# Data Infrastructure Configuration
USE_DUCKDB=false  # Set to true to use DuckDB backend
REDIS_URL=redis://localhost:6379
DUCKDB_PATH=crypto_data.duckdb

# Dashboard Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
EOF

# 6. Run migration
echo "5. Ready to migrate data..."
echo ""
echo "To complete setup:"
echo "1. Copy the provided Python files to infrastructure/ directories"
echo "2. Run: python infrastructure/database/migrate_data.py"
echo "3. Test: python test_duckdb_migration.py"
echo "4. Enable DuckDB: export USE_DUCKDB=true"
echo "5. Run dashboard: streamlit run main.py"