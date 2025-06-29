#!/bin/bash
# Setup script for crypto dashboard infrastructure
# Place in: crypto-dashboard/setup.sh

set -e

echo "🚀 Crypto Dashboard Infrastructure Setup"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs scripts/{__pycache__}
mkdir -p infrastructure/{collectors,message_bus,processors,database,scheduler}/__pycache__
mkdir -p core/__pycache__ applications/dashboard/__pycache__

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration"
fi

# Check Redis
echo "Checking Redis..."
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✓ Redis is running"
    else
        echo "⚠️  Redis is not running. Starting Redis..."
        if command -v redis-server &> /dev/null; then
            redis-server --daemonize yes
        else
            echo "❌ Redis is not installed. Please install Redis or use Docker"
        fi
    fi
else
    echo "⚠️  Redis CLI not found. Please ensure Redis is installed"
fi

# Initialize infrastructure
echo "Initializing infrastructure..."
python scripts/init_infrastructure.py

# Run tests
echo "Running infrastructure tests..."
python scripts/test_infrastructure.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Start services:"
echo "   - Using Docker: docker-compose up -d"
echo "   - Or manually:"
echo "     * redis-server"
echo "     * celery -A infrastructure.scheduler.celery_app worker --beat"
echo "     * python scripts/run_collector.py websocket"
echo "     * python scripts/run_processor.py"
echo "     * streamlit run main.py"
echo ""
echo "3. Access dashboard at http://localhost:8501"
echo "4. Monitor Celery at http://localhost:5555"

