"""
Data Processor Service - Validates and processes incoming data
Place this file in: crypto-dashboard-modular/infrastructure/processors/data_processor.py
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json

from infrastructure.message_bus.events import MessageBus, Event, EventType, create_event, EventHandler
from infrastructure.database.duckdb_schema_dal import get_db_manager, DataInterval

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Data validation error"""
    pass


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Optional[pd.DataFrame] = None


class DataQuality(Enum):
    """Data quality levels"""
    HIGH = "high"       # No issues
    MEDIUM = "medium"   # Minor issues, data usable
    LOW = "low"         # Major issues, data questionable
    INVALID = "invalid" # Data unusable


class DataValidator:
    """Validates incoming OHLCV data"""
    
    def __init__(self):
        self.price_tolerance = 0.5  # 50% price change tolerance
        self.volume_outlier_threshold = 10  # 10x average volume
        
    def validate_ohlcv(self, data: pd.DataFrame, symbol: str, interval: str) -> ValidationResult:
        """Validate OHLCV data"""
        errors = []
        warnings = []
        
        # Basic structure validation
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return ValidationResult(False, errors, warnings)
        
        # Copy data for cleaning
        cleaned_data = data.copy()
        
        # Timestamp validation
        try:
            if not pd.api.types.is_datetime64_any_dtype(cleaned_data['timestamp']):
                cleaned_data['timestamp'] = pd.to_datetime(cleaned_data['timestamp'])
        except Exception as e:
            errors.append(f"Invalid timestamp format: {e}")
            return ValidationResult(False, errors, warnings)
        
        # Check for duplicates
        duplicates = cleaned_data.duplicated(subset=['timestamp']).sum()
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate timestamps, removing...")
            cleaned_data = cleaned_data.drop_duplicates(subset=['timestamp'], keep='last')
        
        # OHLC relationship validation
        invalid_ohlc = (
            (cleaned_data['high'] < cleaned_data['low']) |
            (cleaned_data['open'] > cleaned_data['high']) |
            (cleaned_data['open'] < cleaned_data['low']) |
            (cleaned_data['close'] > cleaned_data['high']) |
            (cleaned_data['close'] < cleaned_data['low'])
        )
        
        if invalid_ohlc.any():
            count = invalid_ohlc.sum()
            errors.append(f"Found {count} candles with invalid OHLC relationships")
            # Try to fix simple cases
            cleaned_data.loc[invalid_ohlc, 'high'] = cleaned_data.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].max(axis=1)
            cleaned_data.loc[invalid_ohlc, 'low'] = cleaned_data.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].min(axis=1)
            warnings.append(f"Attempted to fix {count} invalid OHLC relationships")
        
        # Check for negative values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            negative_count = (cleaned_data[col] < 0).sum()
            if negative_count > 0:
                errors.append(f"Found {negative_count} negative values in {col}")
        
        # Check for missing values
        null_counts = cleaned_data[required_columns].isnull().sum()
        if null_counts.any():
            for col, count in null_counts[null_counts > 0].items():
                warnings.append(f"Found {count} null values in {col}")
                # Forward fill for prices, zero for volume
                if col in ['open', 'high', 'low', 'close']:
                    cleaned_data[col].fillna(method='ffill', inplace=True)
                else:
                    cleaned_data[col].fillna(0, inplace=True)
        
        # Check for extreme price movements
        if len(cleaned_data) > 1:
            price_changes = cleaned_data['close'].pct_change().abs()
            extreme_changes = price_changes > self.price_tolerance
            if extreme_changes.any():
                count = extreme_changes.sum()
                warnings.append(f"Found {count} extreme price movements (>{self.price_tolerance*100}%)")
        
        # Check for volume outliers
        if len(cleaned_data) > 10:
            volume_mean = cleaned_data['volume'].mean()
            volume_outliers = cleaned_data['volume'] > (volume_mean * self.volume_outlier_threshold)
            if volume_outliers.any():
                count = volume_outliers.sum()
                warnings.append(f"Found {count} volume outliers (>{self.volume_outlier_threshold}x average)")
        
        # Check timestamp continuity
        expected_interval = DataInterval(interval).minutes * 60  # seconds
        time_diffs = cleaned_data['timestamp'].diff().dt.total_seconds()
        gaps = time_diffs[time_diffs > expected_interval * 1.5]
        if len(gaps) > 0:
            warnings.append(f"Found {len(gaps)} gaps in timestamp continuity")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, cleaned_data)
    
    def assess_quality(self, validation_result: ValidationResult) -> DataQuality:
        """Assess overall data quality"""
        if not validation_result.is_valid:
            return DataQuality.INVALID
        
        warning_count = len(validation_result.warnings)
        if warning_count == 0:
            return DataQuality.HIGH
        elif warning_count <= 2:
            return DataQuality.MEDIUM
        else:
            return DataQuality.LOW


class DataAggregator:
    """Aggregates data to different timeframes"""
    
    @staticmethod
    def aggregate_ohlcv(
        data: pd.DataFrame,
        source_interval: str,
        target_interval: str
    ) -> pd.DataFrame:
        """Aggregate OHLCV data to a higher timeframe"""
        source_minutes = DataInterval(source_interval).minutes
        target_minutes = DataInterval(target_interval).minutes
        
        if target_minutes <= source_minutes:
            raise ValueError(f"Target interval must be larger than source interval")
        
        # Set timestamp as index
        data = data.set_index('timestamp').sort_index()
        
        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'quote_volume': 'sum',
            'trades': 'sum'
        }
        
        # Resample to target timeframe
        resampled = data.resample(f'{target_minutes}T').agg(agg_rules)
        
        # Remove any rows with all NaN
        resampled = resampled.dropna(how='all')
        
        # Reset index
        resampled = resampled.reset_index()
        
        return resampled


class DataProcessor(EventHandler):
    """
    Main data processor that handles incoming data events
    """
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.db_manager = get_db_manager()
        self.validator = DataValidator()
        self.aggregator = DataAggregator()
        self.processing_stats = {
            'processed': 0,
            'errors': 0,
            'warnings': 0
        }
    
    @property
    def event_types(self) -> List[EventType]:
        """Events this processor handles"""
        return [
            EventType.DATA_FETCH_COMPLETED,
            EventType.PRICE_UPDATE,
            EventType.TRADE_UPDATE
        ]
    
    async def handle(self, event: Event):
        """Process incoming data event"""
        try:
            if event.type == EventType.DATA_FETCH_COMPLETED:
                await self._process_ohlcv_data(event)
            elif event.type == EventType.PRICE_UPDATE:
                await self._process_price_update(event)
            elif event.type == EventType.TRADE_UPDATE:
                await self._process_trade_update(event)
                
            self.processing_stats['processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing event {event.id}: {e}")
            self.processing_stats['errors'] += 1
            
            # Publish error event
            error_event = create_event(
                EventType.ERROR_ALERT,
                source='DataProcessor',
                data={
                    'original_event_id': event.id,
                    'error': str(e),
                    'event_type': event.type.value
                }
            )
            self.message_bus.publish_event(error_event)
    
    async def _process_ohlcv_data(self, event: Event):
        """Process OHLCV data from collectors"""
        data = event.data
        symbol = data.get('symbol')
        interval = data.get('interval')
        
        # Convert data to DataFrame
        if 'dataframe' in data:
            df = pd.DataFrame(data['dataframe'])
        elif 'candles' in data:
            df = pd.DataFrame(data['candles'])
        else:
            raise ValueError("No data found in event")
        
        # Validate data
        validation_result = self.validator.validate_ohlcv(df, symbol, interval)
        
        if not validation_result.is_valid:
            logger.error(f"Validation failed for {symbol} {interval}: {validation_result.errors}")
            return
        
        if validation_result.warnings:
            logger.warning(f"Validation warnings for {symbol} {interval}: {validation_result.warnings}")
            self.processing_stats['warnings'] += len(validation_result.warnings)
        
        # Assess quality
        quality = self.validator.assess_quality(validation_result)
        
        # Store in database
        rows_inserted = self.db_manager.insert_ohlcv_batch(
            validation_result.cleaned_data,
            symbol,
            interval
        )
        
        # Generate aggregations for minute data
        if interval in ['1m', '5m', '15m']:
            await self._generate_aggregations(
                validation_result.cleaned_data,
                symbol,
                interval
            )
        
        # Publish completion event
        completion_event = create_event(
            EventType.DATABASE_UPDATED,
            source='DataProcessor',
            data={
                'symbol': symbol,
                'interval': interval,
                'rows_inserted': rows_inserted,
                'quality': quality.value,
                'warnings': len(validation_result.warnings)
            }
        )
        self.message_bus.publish_event(completion_event)
        
        # Check for anomalies and generate alerts
        await self._check_for_anomalies(validation_result.cleaned_data, symbol, interval)
    
    async def _generate_aggregations(self, data: pd.DataFrame, symbol: str, interval: str):
        """Generate higher timeframe aggregations"""
        # Define aggregation targets based on source interval
        aggregation_map = {
            '1m': ['5m', '15m', '1h'],
            '5m': ['15m', '1h'],
            '15m': ['1h', '4h']
        }
        
        targets = aggregation_map.get(interval, [])
        
        for target_interval in targets:
            try:
                aggregated = self.aggregator.aggregate_ohlcv(
                    data,
                    interval,
                    target_interval
                )
                
                # Store aggregated data
                self.db_manager.insert_ohlcv_batch(
                    aggregated,
                    symbol,
                    target_interval
                )
                
                logger.info(f"Generated {target_interval} aggregation for {symbol} from {interval} data")
                
            except Exception as e:
                logger.error(f"Error generating {target_interval} aggregation: {e}")
    
    async def _process_price_update(self, event: Event):
        """Process real-time price updates"""
        data = event.data
        
        # Update live price in database
        self.db_manager.update_live_price(
            symbol=data['symbol'],
            price=data['price'],
            timestamp=event.timestamp,
            bid=data.get('bid'),
            ask=data.get('ask')
        )
        
        # Check for price alerts
        await self._check_price_alerts(data['symbol'], data['price'])
    
    async def _process_trade_update(self, event: Event):
        """Process real-time trade updates"""
        # For now, just log trades
        # Could aggregate into volume profiles, order flow, etc.
        data = event.data
        logger.debug(f"Trade: {data['symbol']} @ {data['price']} x {data['quantity']}")
    
    async def _check_for_anomalies(self, data: pd.DataFrame, symbol: str, interval: str):
        """Check for anomalies in the data"""
        if len(data) < 10:
            return
        
        # Check for unusual volume
        volume_mean = data['volume'].mean()
        volume_std = data['volume'].std()
        
        # Last candle volume
        last_volume = data['volume'].iloc[-1]
        if last_volume > volume_mean + (3 * volume_std):
            alert_event = create_event(
                EventType.VOLUME_ALERT,
                source='DataProcessor',
                data={
                    'symbol': symbol,
                    'interval': interval,
                    'volume': last_volume,
                    'average_volume': volume_mean,
                    'std_dev': volume_std,
                    'alert_type': 'high_volume'
                }
            )
            self.message_bus.publish_event(alert_event)
        
        # Check for price spikes
        returns = data['close'].pct_change()
        if abs(returns.iloc[-1]) > 0.1:  # 10% move
            alert_event = create_event(
                EventType.PRICE_ALERT,
                source='DataProcessor',
                data={
                    'symbol': symbol,
                    'interval': interval,
                    'price_change': returns.iloc[-1] * 100,
                    'current_price': data['close'].iloc[-1],
                    'alert_type': 'large_movement'
                }
            )
            self.message_bus.publish_event(alert_event)
    
    async def _check_price_alerts(self, symbol: str, price: float):
        """Check if price crosses any alert thresholds"""
        # This would check against user-defined alerts
        # For now, just check major levels
        major_levels = {
            'BTCUSDT': [90000, 95000, 100000, 105000],
            'ETHUSDT': [3000, 3500, 4000, 4500],
            'SOLUSDT': [150, 200, 250, 300]
        }
        
        levels = major_levels.get(symbol, [])
        for level in levels:
            if abs(price - level) / level < 0.001:  # Within 0.1% of level
                alert_event = create_event(
                    EventType.PRICE_ALERT,
                    source='DataProcessor',
                    data={
                        'symbol': symbol,
                        'current_price': price,
                        'level': level,
                        'alert_type': 'near_major_level'
                    }
                )
                self.message_bus.publish_event(alert_event)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            'quality_stats': {
                # Would track quality metrics over time
            }
        }


# Service runner
async def run_processor():
    """Run the data processor service"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize components
    message_bus = MessageBus()
    processor = DataProcessor(message_bus)
    
    # Subscribe to events
    message_bus.subscribe(
        processor.event_types,
        processor.handle,
        consumer_group='data_processors',
        consumer_name='processor_1'
    )
    
    logger.info("Data Processor Service started")
    
    try:
        # Start consuming events
        await message_bus.start_consuming()
    except KeyboardInterrupt:
        logger.info("Shutting down Data Processor...")
        message_bus.stop()


if __name__ == "__main__":
    asyncio.run(run_processor())