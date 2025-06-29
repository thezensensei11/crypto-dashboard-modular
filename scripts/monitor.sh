

echo "🔍 Crypto Dashboard Monitor"
echo "=========================="

# Function to check service
check_service() {
    local service=$1
    local check_command=$2
    
    if eval $check_command &> /dev/null; then
        echo "✅ $service: Running"
        return 0
    else
        echo "❌ $service: Not running"
        return 1
    fi
}

# Check Redis
check_service "Redis" "redis-cli ping"

# Check Celery Worker
check_service "Celery Worker" "pgrep -f 'celery.*worker'"

# Check Celery Beat
check_service "Celery Beat" "pgrep -f 'celery.*beat'"

# Check Database
if [ -f "crypto_data.duckdb" ]; then
    size=$(ls -lh crypto_data.duckdb | awk '{print $5}')
    echo "✅ DuckDB: Exists (Size: $size)"
else
    echo "❌ DuckDB: Not found"
fi

# Check message bus
echo ""
echo "📊 Message Bus Stats:"
if command -v redis-cli &> /dev/null; then
    for stream in price.update candle.closed historical.data; do
        count=$(redis-cli xlen crypto:stream:$stream 2>/dev/null || echo "0")
        echo "  - $stream: $count messages"
    done
fi

# Check recent data
echo ""
echo "📈 Recent Data:"
python3 -c "
import sys
sys.path.append('.')
from infrastructure.database.manager import get_db_manager
from datetime import datetime, timedelta

try:
    db = get_db_manager()
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        latest = db.get_latest_timestamp(symbol, '1m')
        if latest:
            age = datetime.utcnow() - latest
            if age < timedelta(minutes=5):
                status = '🟢'
            elif age < timedelta(hours=1):
                status = '🟡'
            else:
                status = '🔴'
            print(f'  {status} {symbol}: {latest} ({age.total_seconds()//60:.0f} min ago)')
        else:
            print(f'  ⚫ {symbol}: No data')
except Exception as e:
    print(f'  Error: {e}')
"

echo ""
echo "💡 Tips:"
echo "  - View logs: docker-compose logs -f [service-name]"
echo "  - Restart service: docker-compose restart [service-name]"
echo "  - Manual backfill: python scripts/manual_backfill.py --help"