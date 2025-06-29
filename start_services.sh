#!/bin/bash
# Quick start script for crypto dashboard
# Save as: start_services.sh
# Make executable: chmod +x start_services.sh
# Run: ./start_services.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}🚀 Starting Crypto Dashboard Services${NC}"
echo "===================================="

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓ Docker found - using Docker Compose${NC}"
    
    # Check if .env exists
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}Creating .env from .env.example...${NC}"
        cp .env.example .env
        echo -e "${YELLOW}⚠️  Please edit .env with your configuration${NC}"
    fi
    
    # Start services with Docker Compose
    echo -e "\n${BOLD}Starting all services...${NC}"
    docker-compose up -d
    
    # Wait a moment for services to start
    echo -e "\n${BOLD}Waiting for services to initialize...${NC}"
    sleep 5
    
    # Show service status
    echo -e "\n${BOLD}Service Status:${NC}"
    docker-compose ps
    
    echo -e "\n${GREEN}✅ All services started!${NC}"
    echo -e "\n${BOLD}Access points:${NC}"
    echo "  📊 Dashboard: http://localhost:8501"
    echo "  🌻 Celery Flower: http://localhost:5555"
    echo "  🔍 Logs: docker-compose logs -f [service-name]"
    echo ""
    echo -e "${BOLD}To stop all services:${NC} docker-compose down"
    
else
    echo -e "${YELLOW}Docker not found - starting services manually${NC}"
    
    # Check Redis
    if ! command -v redis-cli &> /dev/null; then
        echo -e "${RED}❌ Redis not installed. Please install Redis first.${NC}"
        exit 1
    fi
    
    # Check if Redis is running
    if ! redis-cli ping &> /dev/null; then
        echo -e "${YELLOW}Starting Redis...${NC}"
        redis-server --daemonize yes
        sleep 2
    fi
    echo -e "${GREEN}✓ Redis running${NC}"
    
    # Create log directory
    mkdir -p logs
    
    # Function to start service in background
    start_service() {
        local name=$1
        local command=$2
        local log_file="logs/${name}.log"
        
        echo -e "Starting ${name}..."
        nohup $command > "$log_file" 2>&1 &
        echo $! > "logs/${name}.pid"
        echo -e "${GREEN}✓ ${name} started (PID: $(cat logs/${name}.pid))${NC}"
    }
    
    # Start services
    start_service "data-processor" "python scripts/run_processor.py"
    sleep 2
    
    start_service "websocket-collector" "python scripts/run_collector.py websocket"
    sleep 2
    
    start_service "rest-collector" "python scripts/run_collector.py rest"
    sleep 2
    
    start_service "celery-worker" "celery -A infrastructure.scheduler.celery_app worker --loglevel=info"
    sleep 2
    
    start_service "celery-beat" "celery -A infrastructure.scheduler.celery_app beat --loglevel=info"
    sleep 2
    
    # Start Streamlit in foreground
    echo -e "\n${BOLD}Starting Dashboard...${NC}"
    echo -e "${GREEN}✅ All background services started!${NC}"
    echo -e "\n${BOLD}Access points:${NC}"
    echo "  📊 Dashboard: http://localhost:8501"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop the dashboard${NC}"
    echo -e "${YELLOW}Run ./stop_services.sh to stop all services${NC}"
    echo ""
    
    # Run Streamlit
    streamlit run main.py
fi
