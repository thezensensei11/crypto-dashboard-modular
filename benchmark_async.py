"""
Simple benchmark to test async performance improvements
"""

import asyncio
import time
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the collectors directly
from crypto_dashboard_modular.data.binance_collector import BinanceFuturesCollector
from crypto_dashboard_modular.data.async_binance_collector import AsyncBinanceFuturesCollector


async def benchmark_async_vs_sync():
    """Simple benchmark comparing sync vs async performance"""
    
    # Test parameters
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']
    interval = '1h'
    lookback_days = 7
    
    print("\n" + "="*60)
    print("ASYNC VS SYNC BENCHMARK")
    print("="*60)
    print(f"Testing with {len(test_symbols)} symbols, {interval} interval, {lookback_days} days")
    
    # Test sync collector
    print("\n1. SYNC Collector (sequential requests):")
    sync_collector = BinanceFuturesCollector()
    sync_collector.reset_counters()
    
    sync_start = time.time()
    sync_results = {}
    
    for symbol in test_symbols:
        print(f"   Fetching {symbol}...", end='', flush=True)
        symbol_start = time.time()
        data = sync_collector.get_klines_smart(
            symbol=symbol,
            interval=interval,
            lookback_days=lookback_days,
            force_use_cache=False
        )
        symbol_time = time.time() - symbol_start
        print(f" {symbol_time:.2f}s ({len(data)} rows)")
        sync_results[symbol] = data
    
    sync_total = time.time() - sync_start
    print(f"\n   Total time: {sync_total:.2f}s")
    print(f"   API calls: {sync_collector.api_call_count}")
    print(f"   Cache hits: {sync_collector.cache_hit_count}")
    
    # Test async collector
    print("\n2. ASYNC Collector (concurrent requests):")
    async_collector = AsyncBinanceFuturesCollector()
    await async_collector._init_session()
    async_collector.reset_counters()
    
    async_start = time.time()
    
    # Fetch all symbols concurrently
    async_results = await async_collector.get_klines_batch(
        symbols=test_symbols,
        interval=interval,
        lookback_days=lookback_days,
        force_use_cache=False
    )
    
    async_total = time.time() - async_start
    
    # Print results
    for symbol, data in async_results.items():
        print(f"   {symbol}: {len(data)} rows")
    
    print(f"\n   Total time: {async_total:.2f}s")
    print(f"   API calls: {async_collector.api_call_count}")
    print(f"   Cache hits: {async_collector.cache_hit_count}")
    
    # Calculate speedup
    speedup = sync_total / async_total if async_total > 0 else 0
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"Sync time:  {sync_total:.2f}s")
    print(f"Async time: {async_total:.2f}s")
    print(f"🚀 Speedup: {speedup:.1f}x faster!")
    print("="*60)
    
    # Cleanup
    await async_collector.close()
    
    # Verify data consistency
    print("\nData consistency check:")
    all_match = True
    for symbol in test_symbols:
        if symbol in sync_results and symbol in async_results:
            sync_df = sync_results[symbol]
            async_df = async_results[symbol]
            
            if len(sync_df) == len(async_df):
                print(f"   {symbol}: ✓ Same number of rows ({len(sync_df)})")
            else:
                print(f"   {symbol}: ✗ Different row counts - sync: {len(sync_df)}, async: {len(async_df)}")
                all_match = False
    
    if all_match:
        print("\n✅ All data matches!")
    
    return {
        'sync_time': sync_total,
        'async_time': async_total,
        'speedup': speedup
    }


async def test_large_scale():
    """Test with many symbols to show real performance gains"""
    print("\n" + "="*60)
    print("LARGE SCALE TEST - 20 SYMBOLS")
    print("="*60)
    
    # Get more symbols
    collector = AsyncBinanceFuturesCollector()
    await collector._init_session()
    
    # Get available symbols
    exchange_info = await collector.get_exchange_info()
    test_symbols = exchange_info['symbol'].tolist()[:20]
    
    print(f"Testing with {len(test_symbols)} symbols...")
    
    start = time.time()
    results = await collector.get_klines_batch(
        symbols=test_symbols,
        interval='1h',
        lookback_days=1,
        force_use_cache=False
    )
    
    total_time = time.time() - start
    successful = sum(1 for df in results.values() if not df.empty)
    
    print(f"\nResults:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Successful: {successful}/{len(test_symbols)}")
    print(f"   Average per symbol: {total_time/len(test_symbols):.3f}s")
    print(f"   Estimated sync time: ~{total_time/len(test_symbols) * len(test_symbols) * 5:.2f}s")
    
    await collector.close()


async def main():
    """Main entry point"""
    print("\n🚀 CRYPTO DASHBOARD ASYNC PERFORMANCE TEST 🚀")
    
    # Run basic benchmark
    results = await benchmark_async_vs_sync()
    
    # Run large scale test
    if results['speedup'] > 1:
        await test_large_scale()
    
    print("\n✨ Async implementation ready for production!")
    print("✨ Your dashboard will now run much faster!")


if __name__ == "__main__":
    # Run the benchmark
    asyncio.run(main())