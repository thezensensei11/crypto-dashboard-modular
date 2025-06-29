#!/usr/bin/env python3
"""
Fixed Data Processor Script with Proper DuckDB Connection Management
Save as: scripts/run_processor.py (replace the existing one)

This version includes:
- Proper connection management
- Periodic connection refresh
- Signal handling for graceful shutdown
- Automatic cleanup on exit
- Better error handling
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
import signal
import atexit
from datetime import datetime, timezone
from typing import Optional

# Import your existing DataProcessor
from infrastructure.processors.data_processor import DataProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('processor_debug.log')
    ]
)
logger = logging.getLogger(__name__)


class ManagedDataProcessor:
    """
    Wrapper for DataProcessor with proper connection management
    """
    
    def __init__(self):
        self.processor: Optional[DataProcessor] = None
        self.running = False
        self._shutdown_event = asyncio.Event()
        self._connection_refresh_interval = 30  # seconds
        self._stats_log_interval = 60  # seconds
        self._last_refresh = datetime.now(timezone.utc)
        
    async def start(self):
        """Start the processor with managed connections"""
        logger.info("Starting managed data processor...")
        
        try:
            # Create processor instance
            self.processor = DataProcessor()
            self.running = True
            
            # Start the processor
            await self.processor.start()
            logger.info("Data processor started successfully")
            
            # Create tasks for maintenance
            maintenance_task = asyncio.create_task(self._maintenance_loop())
            stats_task = asyncio.create_task(self._stats_loop())
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
            # Cancel maintenance tasks
            maintenance_task.cancel()
            stats_task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(
                maintenance_task, 
                stats_task, 
                return_exceptions=True
            )
            
        except Exception as e:
            logger.error(f"Processor error: {e}", exc_info=True)
            raise
        finally:
            await self.cleanup()
    
    async def _maintenance_loop(self):
        """Periodic maintenance tasks"""
        while self.running:
            try:
                # Wait for interval or shutdown
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._connection_refresh_interval
                )
            except asyncio.TimeoutError:
                # Time to refresh connections
                await self._refresh_connections()
    
    async def _stats_loop(self):
        """Periodic stats logging"""
        while self.running:
            try:
                # Wait for interval or shutdown
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._stats_log_interval
                )
            except asyncio.TimeoutError:
                # Log stats
                if self.processor:
                    stats = self.processor.get_stats()
                    logger.info(f"Processor stats: {stats}")
    
    async def _refresh_connections(self):
        """Refresh database connections to prevent locks"""
        try:
            logger.info("Refreshing database connections...")
            
            if self.processor and hasattr(self.processor, 'db_manager'):
                # Close existing connections
                if hasattr(self.processor.db_manager, 'close'):
                    self.processor.db_manager.close()
                
                # For connection pools
                if hasattr(self.processor.db_manager, '_pool'):
                    self.processor.db_manager._pool.close_all()
                
                # For thread-local connections
                if hasattr(self.processor.db_manager, '_thread_local'):
                    if hasattr(self.processor.db_manager._thread_local, 'conn'):
                        try:
                            self.processor.db_manager._thread_local.conn.close()
                        except:
                            pass
                        delattr(self.processor.db_manager._thread_local, 'conn')
                
                self._last_refresh = datetime.now(timezone.utc)
                logger.info("Database connections refreshed successfully")
                
        except Exception as e:
            logger.error(f"Error refreshing connections: {e}")
    
    async def cleanup(self):
        """Clean shutdown with proper resource cleanup"""
        logger.info("Cleaning up data processor...")
        
        try:
            if self.processor:
                # Stop the processor
                logger.info("Stopping processor...")
                await self.processor.stop()
                
                # Force flush any remaining data
                if hasattr(self.processor, '_flush_all_buffers'):
                    await self.processor._flush_all_buffers()
                
                # Close database connections
                await self._refresh_connections()
                
                # Additional cleanup for any remaining connections
                if hasattr(self.processor, 'db_manager'):
                    db_manager = self.processor.db_manager
                    
                    # Try multiple cleanup methods
                    cleanup_methods = [
                        'close',
                        'close_all',
                        'disconnect',
                        'cleanup'
                    ]
                    
                    for method in cleanup_methods:
                        if hasattr(db_manager, method):
                            try:
                                getattr(db_manager, method)()
                                logger.info(f"Called {method} on db_manager")
                            except Exception as e:
                                logger.debug(f"Error calling {method}: {e}")
                
                # Clear the processor reference
                self.processor = None
                
            logger.info("Cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def shutdown(self):
        """Signal shutdown"""
        logger.info("Shutdown requested...")
        self.running = False
        self._shutdown_event.set()


# Global processor instance for signal handlers
_processor_instance: Optional[ManagedDataProcessor] = None


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    
    if _processor_instance:
        # Create task to handle async shutdown
        asyncio.create_task(_processor_instance.shutdown())


def cleanup_on_exit():
    """Cleanup function for atexit"""
    logger.info("Exit handler called")
    
    # Try to clean up any remaining connections
    try:
        from infrastructure.database.duckdb_manager import DuckDBConnectionPool
        pool = DuckDBConnectionPool()
        pool.close_all()
        logger.info("Closed DuckDB connection pool")
    except Exception as e:
        logger.debug(f"Error closing connection pool: {e}")
    
    # Try to close any singleton instances
    try:
        from infrastructure.database.manager import get_db_manager
        db_manager = get_db_manager()
        if hasattr(db_manager, 'close'):
            db_manager.close()
        logger.info("Closed database manager")
    except Exception as e:
        logger.debug(f"Error closing database manager: {e}")


async def run_data_processor():
    """Main entry point for the data processor"""
    global _processor_instance
    
    # Register cleanup handlers
    atexit.register(cleanup_on_exit)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run processor
    _processor_instance = ManagedDataProcessor()
    
    try:
        await _processor_instance.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        # Ensure cleanup happens
        if _processor_instance:
            await _processor_instance.cleanup()
        
        # Clear the global reference
        _processor_instance = None


def main():
    """Main function with proper async cleanup"""
    logger.info("=" * 50)
    logger.info("Starting Crypto Data Processor")
    logger.info(f"Time: {datetime.now(timezone.utc)}")
    logger.info("=" * 50)
    
    try:
        # Run the async processor
        asyncio.run(run_data_processor())
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        
    finally:
        # Final cleanup attempt
        logger.info("Final cleanup...")
        
        # Force close any remaining event loops
        try:
            loop = asyncio.get_event_loop()
            if loop and not loop.is_closed():
                # Cancel all remaining tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                
                # Run until all tasks are cancelled
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
                
                # Close the loop
                loop.close()
                
        except Exception as e:
            logger.debug(f"Error closing event loop: {e}")
        
        logger.info("Process terminated")


if __name__ == "__main__":
    main()