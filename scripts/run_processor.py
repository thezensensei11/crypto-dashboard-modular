
"""
Script to run data processor
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

sys.path.append(str(Path(__file__).parent.parent))

from infrastructure.processors.data_processor import DataProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def run_data_processor():
    """Run data processor"""
    processor = DataProcessor()
    
    try:
        await processor.start()
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            stats = processor.get_stats()
            logging.info(f"Processor stats: {stats}")
            
    except KeyboardInterrupt:
        logging.info("Shutting down data processor...")
        await processor.stop()


if __name__ == "__main__":
    asyncio.run(run_data_processor())

