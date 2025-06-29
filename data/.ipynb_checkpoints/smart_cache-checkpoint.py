import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple, Optional
import logging


class SmartDataManager:
    """
    Intelligent data manager that:
    1. Tracks what data exists locally
    2. Identifies gaps that need to be fetched
    3. Merges new data without duplicates
    4. Provides fast access for calculations
    
    IMPORTANT: All timestamps are handled in UTC to match Binance API
    """
    
    # REMOVED API delay buffer - we want the latest data!
    API_DELAY_BUFFER_MINUTES = 0  # No delay - fetch latest available data
    
    def __init__(self, cache_dir: str = "smart_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Create metadata file to track what we have
        self.metadata_file = self.cache_dir / "data_inventory.json"
        self.metadata = self._load_metadata()
    
    def _get_utc_now(self) -> datetime:
        """Get current time in UTC (timezone-aware)"""
        return datetime.now(timezone.utc)
    
    def _ensure_utc(self, dt: datetime) -> datetime:
        """Ensure datetime is UTC (convert if necessary)"""
        if dt.tzinfo is None:
            # Naive datetime - assume it's UTC
            self.logger.warning(f"Naive datetime {dt} assumed to be UTC")
            return dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            # Convert to UTC
            return dt.astimezone(timezone.utc)
        return dt
        
    def _load_metadata(self) -> dict:
        """Load or create metadata tracking what data we have"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "symbols": {},  # symbol -> {interval -> {first_timestamp, last_timestamp, row_count}}
            "last_updated": None
        }
    
    def _save_metadata(self):
        """Save metadata to file"""
        self.metadata["last_updated"] = self._get_utc_now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_data_inventory(self) -> dict:
        """Get a summary of what data we have cached"""
        inventory = {
            "total_symbols": len(self.metadata["symbols"]),
            "total_files": 0,
            "total_rows": 0,
            "details": {}
        }
        
        for symbol, intervals in self.metadata["symbols"].items():
            symbol_info = {
                "intervals": {},
                "total_rows": 0
            }
            for interval, info in intervals.items():
                symbol_info["intervals"][interval] = {
                    "rows": info["row_count"],
                    "start": info["first_timestamp"],
                    "end": info["last_timestamp"]
                }
                symbol_info["total_rows"] += info["row_count"]
                inventory["total_files"] += 1
            
            inventory["details"][symbol] = symbol_info
            inventory["total_rows"] += symbol_info["total_rows"]
        
        return inventory
    
    def identify_data_gaps(self, symbol: str, interval: str, 
                          start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime]]:
        """
        Identify what date ranges need to be fetched
        Now with ZERO delay - fetch all available complete candles
        
        IMPORTANT: Timestamps represent candle OPEN times!
        """
        # Check if we have any data for this symbol/interval
        if symbol not in self.metadata["symbols"]:
            self.logger.info(f"No data for {symbol} {interval}, need full range")
            return [(start_date, end_date)]
        
        if interval not in self.metadata["symbols"][symbol]:
            self.logger.info(f"No data for {symbol} {interval} interval, need full range")
            return [(start_date, end_date)]
        
        # Get existing data info
        existing = self.metadata["symbols"][symbol][interval]
        existing_start = datetime.fromisoformat(existing["first_timestamp"])
        existing_end = datetime.fromisoformat(existing["last_timestamp"])  # This is the OPEN time of the last candle
        
        # Calculate when the last candle actually closes
        interval_minutes = self._interval_to_minutes(interval)
        last_candle_close_time = existing_end + timedelta(minutes=interval_minutes)
        
        self.logger.info(f"{symbol} {interval}: Have data from {existing_start} to {existing_end}")
        self.logger.info(f"Last candle opens at {existing_end}, closes at {last_candle_close_time}")
        self.logger.info(f"Requested: {start_date} to {end_date}")
        
        gaps = []
        current_time = self._get_utc_now()
        
        # Need earlier data? Only if requested start is before our cache
        if start_date < existing_start:
            gap_end = min(existing_start - timedelta(minutes=interval_minutes), end_date)
            gaps.append((start_date, gap_end))
            self.logger.info(f"Need earlier data: {start_date} to {gap_end}")
        
        # Need later data? Check if any complete candles are available after our last one
        next_candle_start = last_candle_close_time
        
        # Calculate the theoretical last complete candle at current time
        # This is the most recent candle that has FINISHED
        last_possible_complete_candle_start = self._get_last_complete_candle_start(current_time, interval)
        
        self.logger.info(f"Current time: {current_time}")
        self.logger.info(f"Last possible complete candle starts at: {last_possible_complete_candle_start}")
        
        # If there are complete candles we don't have, fetch them
        if last_possible_complete_candle_start >= next_candle_start:
            # We have new complete candles to fetch!
            fetch_end = min(last_possible_complete_candle_start, end_date)
            gaps.append((next_candle_start, fetch_end))
            
            # Calculate expected candles
            expected_candles = int((fetch_end - next_candle_start).total_seconds() / 60 / interval_minutes) + 1
            self.logger.info(f"Need {expected_candles} new complete candles from {next_candle_start} to {fetch_end}")
        else:
            # No new complete candles yet
            time_until_next = (next_candle_start - current_time).total_seconds() / 60
            self.logger.info(f"No new complete candles yet. Next candle completes in {time_until_next:.1f} minutes")
        
        # If requested range is entirely within existing range, no gaps
        if not gaps and start_date >= existing_start and end_date <= existing_end:
            self.logger.info("All requested data already cached!")
        
        self.logger.info(f"Total gaps to fetch: {len(gaps)}")
        return gaps
    
    def _get_last_complete_candle_start(self, current_time: datetime, interval: str) -> datetime:
        """
        Calculate the start time of the last COMPLETE candle
        A candle is complete when current_time >= candle_close_time
        """
        interval_minutes = self._interval_to_minutes(interval)
        
        # For minute-based intervals
        if interval in ['1m', '3m', '5m', '15m', '30m']:
            # Calculate minutes since midnight
            minutes_since_midnight = current_time.hour * 60 + current_time.minute
            
            # Find the last candle boundary that has COMPLETED
            # A candle starting at X completes at X + interval_minutes
            # So we need to find the latest X where X + interval_minutes <= current_time
            
            # Round down to candle boundary
            candle_boundary_minutes = (minutes_since_midnight // interval_minutes) * interval_minutes
            candle_start = current_time.replace(
                hour=candle_boundary_minutes // 60,
                minute=candle_boundary_minutes % 60,
                second=0,
                microsecond=0
            )
            
            # Check if this candle is actually complete
            candle_end = candle_start + timedelta(minutes=interval_minutes)
            if candle_end > current_time:
                # This candle is still ongoing, go back one interval
                candle_start = candle_start - timedelta(minutes=interval_minutes)
                
        # For hour-based intervals
        elif interval in ['1h', '2h', '4h', '6h', '8h', '12h']:
            hours_in_interval = interval_minutes // 60
            
            # Round down to hour boundary
            candle_boundary_hour = (current_time.hour // hours_in_interval) * hours_in_interval
            candle_start = current_time.replace(
                hour=candle_boundary_hour,
                minute=0,
                second=0,
                microsecond=0
            )
            
            # Check if this candle is actually complete
            candle_end = candle_start + timedelta(minutes=interval_minutes)
            if candle_end > current_time:
                # This candle is still ongoing, go back one interval
                candle_start = candle_start - timedelta(minutes=interval_minutes)
                
        # For daily intervals
        elif interval == '1d':
            # Daily candles start at 00:00 UTC
            candle_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Check if today's candle is complete
            candle_end = candle_start + timedelta(days=1)
            if candle_end > current_time:
                # Today's candle is still ongoing, use yesterday's
                candle_start = candle_start - timedelta(days=1)
        else:
            # Default fallback
            candle_start = current_time - timedelta(minutes=interval_minutes)
        
        self.logger.debug(f"Last complete candle for {interval} at {current_time}: starts at {candle_start}")
        return candle_start
    
    def get_cache_path(self, symbol: str, interval: str) -> Path:
        """Get the file path for cached data"""
        symbol_dir = self.cache_dir / symbol
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir / f"{interval}.parquet"
    
    def save_data(self, symbol: str, interval: str, data: pd.DataFrame):
        """
        Save data to cache with deduplication and validation
        Uses Parquet format for efficiency
        
        IMPORTANT: Only saves COMPLETE candles, not ongoing ones
        """
        if data.empty:
            return
        
        # Filter out any ongoing candles (where close_time > current time)
        current_time = self._get_utc_now()
        if 'close_time' in data.columns:
            # Calculate close_time if not present
            if data['close_time'].isna().any():
                interval_minutes = self._interval_to_minutes(interval)
                data['close_time'] = data['timestamp'] + timedelta(minutes=interval_minutes)
            
            # Remove any candles that haven't closed yet
            complete_mask = data['close_time'] <= current_time
            if not complete_mask.all():
                self.logger.info(f"Filtering out {(~complete_mask).sum()} ongoing candles")
                data = data[complete_mask].copy()
        else:
            # If no close_time column, calculate it
            interval_minutes = self._interval_to_minutes(interval)
            data['close_time'] = data['timestamp'] + timedelta(minutes=interval_minutes)
            
            # Filter
            complete_mask = data['close_time'] <= current_time
            if not complete_mask.all():
                self.logger.info(f"Filtering out {(~complete_mask).sum()} ongoing candles")
                data = data[complete_mask].copy()
        
        if data.empty:
            self.logger.warning("No complete candles to save")
            return
        
        cache_path = self.get_cache_path(symbol, interval)
        
        # If existing data exists, merge
        if cache_path.exists():
            existing_data = pd.read_parquet(cache_path)
            
            # Combine and remove duplicates
            combined = pd.concat([existing_data, data], ignore_index=True)
            combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
            combined = combined.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Merged data for {symbol} {interval}: "
                           f"{len(existing_data)} + {len(data)} -> {len(combined)} rows")
            
            data = combined
        else:
            # Ensure sorted and no duplicates
            data = data.drop_duplicates(subset=['timestamp'], keep='last')
            data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Save to parquet (much faster than pickle for DataFrames)
        data.to_parquet(cache_path, index=False, compression='snappy')
        
        # Update metadata
        if symbol not in self.metadata["symbols"]:
            self.metadata["symbols"][symbol] = {}
        
        # Store metadata with clear understanding of what timestamps mean
        last_candle_open = data['timestamp'].max()
        last_candle_close = last_candle_open + timedelta(minutes=self._interval_to_minutes(interval))
        
        self.metadata["symbols"][symbol][interval] = {
            "first_timestamp": data['timestamp'].min().isoformat(),  # OPEN time of first candle
            "last_timestamp": last_candle_open.isoformat(),   # OPEN time of last candle
            "last_candle_closes_at": last_candle_close.isoformat(),
            "row_count": len(data),
            "file_size_mb": cache_path.stat().st_size / (1024 * 1024),
            "last_updated": current_time.isoformat(),
            "fetched_at": current_time.isoformat()  # When we actually fetched this data
        }
        
        self._save_metadata()
        self.logger.info(f"Saved {len(data)} rows for {symbol} {interval}")
    
    def load_data(self, symbol: str, interval: str, 
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Load cached data for a symbol/interval
        Optionally filter by date range
        """
        cache_path = self.get_cache_path(symbol, interval)
        
        if not cache_path.exists():
            self.logger.warning(f"No cached data for {symbol} {interval}")
            return None
        
        try:
            data = pd.read_parquet(cache_path)
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Filter by date range if specified
            if start_date:
                data = data[data['timestamp'] >= start_date]
            if end_date:
                data = data[data['timestamp'] <= end_date]
            
            self.logger.info(f"Loaded {len(data)} rows for {symbol} {interval}")
            return data.reset_index(drop=True)
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol} {interval}: {e}")
            return None
    
    def estimate_time_saved(self, gaps_to_fetch: List[Tuple[datetime, datetime]], 
                           total_range: Tuple[datetime, datetime],
                           interval: str) -> dict:
        """
        Estimate time saved by using cache
        """
        # Estimate based on interval (rough API response times)
        api_time_per_request = {
            '1m': 0.3, '5m': 0.3, '15m': 0.3, '30m': 0.3,
            '1h': 0.3, '4h': 0.4, '1d': 0.5
        }.get(interval, 0.3)
        
        # Calculate total time range in minutes
        total_minutes = (total_range[1] - total_range[0]).total_seconds() / 60
        
        # Calculate gap time
        gap_minutes = sum((end - start).total_seconds() / 60 for start, end in gaps_to_fetch) if gaps_to_fetch else 0
        
        # Estimate number of API calls needed (1500 candles per call)
        interval_minutes = self._interval_to_minutes(interval)
        total_candles = total_minutes / interval_minutes
        gap_candles = gap_minutes / interval_minutes if gap_minutes > 0 else 0
        
        total_api_calls = max(1, int(total_candles / 1500))
        gap_api_calls = max(0, int(gap_candles / 1500))  # Can be 0 if no gaps
        
        # Estimate times
        time_without_cache = total_api_calls * api_time_per_request
        time_with_cache = gap_api_calls * api_time_per_request
        time_saved = time_without_cache - time_with_cache
        
        # Calculate percentage cached
        percent_cached = ((total_minutes - gap_minutes) / total_minutes * 100) if total_minutes > 0 else 100
        
        return {
            "time_without_cache": time_without_cache,
            "time_with_cache": time_with_cache,
            "time_saved": time_saved,
            "percent_cached": percent_cached,
            "total_candles": int(total_candles),
            "gap_candles": int(gap_candles)
        }
    
    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes"""
        mapping = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return mapping.get(interval, 60)
    
    def clear_all_cache(self):
        """Nuclear option - delete all cached data"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.metadata = {"symbols": {}, "last_updated": None}
            self._save_metadata()
            self.logger.info("Cleared all cached data")
    
    def verify_data_integrity(self, symbol: str, interval: str) -> dict:
        """
        Verify data integrity for a symbol/interval
        Check for gaps, duplicates, etc.
        """
        data = self.load_data(symbol, interval)
        if data is None or data.empty:
            return {"status": "no_data"}
        
        # Check for duplicates
        duplicates = data.duplicated(subset=['timestamp']).sum()
        
        # Check for gaps
        interval_minutes = self._interval_to_minutes(interval)
        expected_diff = pd.Timedelta(minutes=interval_minutes)
        
        time_diffs = data['timestamp'].diff()
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        
        return {
            "status": "ok",
            "rows": len(data),
            "duplicates": duplicates,
            "gaps": len(gaps),
            "first_timestamp": data['timestamp'].min(),
            "last_timestamp": data['timestamp'].max(),
            "integrity": "good" if duplicates == 0 and len(gaps) == 0 else "issues"
        }
    
    def get_latest_timestamp(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get the latest timestamp for a symbol/interval from cache
        
        Returns the OPEN time of the last candle we have.
        The candle is complete at (timestamp + interval).
        """
        if symbol in self.metadata["symbols"] and interval in self.metadata["symbols"][symbol]:
            try:
                ts_str = self.metadata["symbols"][symbol][interval]["last_timestamp"]
                ts = datetime.fromisoformat(ts_str)
                
                # Sanity check - make sure timestamp is reasonable
                current_time = self._get_utc_now()
                if ts > current_time:
                    self.logger.warning(f"Cached timestamp {ts} is in the future! Current time: {current_time}")
                
                return ts
            except Exception as e:
                self.logger.error(f"Error parsing timestamp for {symbol} {interval}: {e}")
                return None
        return None
    
    def get_last_complete_candle_time(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get the close time of the last candle we have cached"""
        if symbol in self.metadata["symbols"] and interval in self.metadata["symbols"][symbol]:
            metadata = self.metadata["symbols"][symbol][interval]
            # Try new format first
            if "last_candle_closes_at" in metadata:
                return datetime.fromisoformat(metadata["last_candle_closes_at"])
            # Fallback: calculate from last_timestamp
            last_open = datetime.fromisoformat(metadata["last_timestamp"])
            return last_open + timedelta(minutes=self._interval_to_minutes(interval))
        return None
    
    def get_data_freshness(self, symbol: str, interval: str) -> Optional[timedelta]:
        """Get how fresh the data is (time since last candle closed)"""
        last_close = self.get_last_complete_candle_time(symbol, interval)
        if last_close is None:
            return None
        
        # Ensure timezone awareness
        if last_close.tzinfo is None:
            last_close = last_close.replace(tzinfo=timezone.utc)
        
        current_time = self._get_utc_now()
        return current_time - last_close
    
    def get_fetch_time(self, symbol: str, interval: str) -> Optional[datetime]:
        """Get when data was last fetched from API"""
        if symbol in self.metadata["symbols"] and interval in self.metadata["symbols"][symbol]:
            metadata = self.metadata["symbols"][symbol][interval]
            if "fetched_at" in metadata:
                return datetime.fromisoformat(metadata["fetched_at"])
            # Fallback to last_updated
            elif "last_updated" in metadata:
                return datetime.fromisoformat(metadata["last_updated"])
        return None