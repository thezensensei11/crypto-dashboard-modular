# Save as: stop_services.sh
#!/bin/bash
# Stop all services
# Make executable: chmod +x stop_services.sh

echo "🛑 Stopping Crypto Dashboard Services"
echo "===================================="

# Check if using Docker
if [ -f "docker-compose.yml" ] && docker-compose ps &> /dev/null; then
    echo "Stopping Docker services..."
    docker-compose down
    echo "✓ Docker services stopped"
else
    # Stop manual services
    echo "Stopping manual services..."
    
    # Kill processes by PID files
    for pid_file in logs/*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            service=$(basename "$pid_file" .pid)
            if kill -0 "$pid" 2>/dev/null; then
                echo "Stopping $service (PID: $pid)..."
                kill "$pid"
                rm "$pid_file"
            fi
        fi
    done
    
    # Stop any remaining Celery processes
    pkill -f "celery.*infrastructure.scheduler" || true
    
    # Stop Streamlit
    pkill -f "streamlit run" || true
    
    echo "✓ All services stopped"
fi

echo "Done!"