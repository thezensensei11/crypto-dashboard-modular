"""
Infrastructure initialization script
Place in: crypto-dashboard/scripts/init_infrastructure.py
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import get_settings
from infrastructure.database.manager import get_db_manager
from infrastructure.message_bus.bus import get_message_bus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def initialize_infrastructure():
    """Initialize all infrastructure components"""
    logger.info("Initializing crypto data infrastructure...")
    
    settings = get_settings()
    
    # 1. Initialize database
    logger.info("Initializing DuckDB...")
    try:
        db_manager = get_db_manager()
        stats = db_manager.get_data_stats()
        logger.info(f"Database initialized. Stats: {stats}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False
    
    # 2. Test Redis connection
    logger.info("Testing Redis connection...")
    try:
        bus = await get_message_bus()
        await bus.disconnect()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        logger.error("Make sure Redis is running!")
        return False
    
    # 3. Create default configuration files if they don't exist
    config_files = {
        "universe_config.json": [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT"
        ],
        "columns_config.json": [
            "symbol", "close", "volume", "price_change_24h", "price_change_pct_24h",
            "high_24h", "low_24h", "volume_24h"
        ]
    }
    
    for filename, default_content in config_files.items():
        filepath = Path(filename)
        if not filepath.exists():
            import json
            with open(filepath, 'w') as f:
                json.dump(default_content, f, indent=2)
            logger.info(f"Created {filename}")
    
    logger.info("Infrastructure initialization complete!")
    return True


