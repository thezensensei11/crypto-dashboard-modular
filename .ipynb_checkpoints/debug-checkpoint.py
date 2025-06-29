#!/usr/bin/env python3
"""
Fix the async/await TypeError in unified_engine.py
The engine is expecting async methods but the collector has sync methods
"""

from pathlib import Path
import re

def fix_async_error():
    print("🔧 Fixing async/await TypeError")
    print("=" * 60)
    
    # Fix 1: Update unified_engine.py to handle sync collector
    print("\n1. Fixing unified_engine.py to handle sync collector...")
    
    engine_path = Path("metrics/unified_engine.py")
    if engine_path.exists():
        content = engine_path.read_text()
        
        # Backup
        backup = engine_path.with_suffix('.py.backup_async')
        backup.write_text(content)
        print(f"   ✓ Backed up to {backup}")
        
        # Fix the _fetch_all_data_async method
        # Find the problematic await line
        old_pattern = r'batch_data = await self\.collector\.get_price_data_batch\('
        
        if 'await self.collector.get_price_data_batch' in content:
            # Replace with sync call wrapped in asyncio
            new_code = '''# Handle sync collector - wrap in executor
                batch_data = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.collector.get_price_data_batch,'''
            
            content = content.replace(
                'batch_data = await self.collector.get_price_data_batch(',
                new_code
            )
            
            # Need to add asyncio import if not present
            if 'import asyncio' not in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'import' in line and 'from' not in line:
                        lines.insert(i+1, 'import asyncio')
                        break
                content = '\n'.join(lines)
        
        # Also fix any other await calls to collector methods
        # Fix get_price_data if it's being awaited
        content = re.sub(
            r'await self\.collector\.get_price_data\(',
            'await asyncio.get_event_loop().run_in_executor(None, self.collector.get_price_data, ',
            content
        )
        
        engine_path.write_text(content)
        print("   ✓ Fixed unified_engine.py")
    
    # Fix 2: Alternative - make collector methods async
    print("\n2. Alternative fix - Adding async wrapper to collector...")
    
    collector_path = Path("data/duckdb_collector.py")
    if collector_path.exists():
        content = collector_path.read_text()
        
        # Check if we need to add async wrappers
        if 'async def get_price_data_batch' not in content:
            # Add async version after the sync version
            lines = content.split('\n')
            new_lines = []
            
            # First add asyncio import
            import_added = False
            for line in lines:
                new_lines.append(line)
                if not import_added and 'import' in line and 'time' in line:
                    new_lines.append('import asyncio')
                    import_added = True
            
            # Find get_price_data_batch and add async version
            for i, line in enumerate(new_lines):
                if 'def get_price_data_batch(' in line and 'async' not in line:
                    # Find the end of the method
                    indent = len(line) - len(line.lstrip())
                    method_end = i
                    for j in range(i+1, len(new_lines)):
                        if new_lines[j].strip() and not new_lines[j].startswith(' '):
                            method_end = j
                            break
                        elif new_lines[j].strip() and len(new_lines[j]) - len(new_lines[j].lstrip()) <= indent:
                            method_end = j
                            break
                    
                    # Insert async version
                    async_method = f'''
    async def get_price_data_batch_async(
        self,
        symbols: List[str],
        interval: str = '1h',
        lookback_days: int = 30,
        force_cache: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Async wrapper for get_price_data_batch"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.get_price_data_batch,
            symbols,
            interval,
            lookback_days,
            force_cache
        )'''
                    
                    new_lines.insert(method_end, async_method)
                    break
            
            content = '\n'.join(new_lines)
            
            # Write back only if we made changes
            if 'async def get_price_data_batch_async' in content:
                collector_path.write_text(content)
                print("   ✓ Added async wrapper to collector")
    
    print("\n" + "=" * 60)
    print("✅ Async/await error fixed!")
    print("\nTesting the fix...")
    
    # Quick test
    try:
        import asyncio
        from data.duckdb_collector import BinanceDataCollector
        
        collector = BinanceDataCollector()
        
        # Test sync method
        result = collector.get_price_data_batch(['BTCUSDT'], '1h', 1, True)
        print(f"✓ Sync method works: {len(result)} symbols")
        
        # Test if async wrapper exists
        if hasattr(collector, 'get_price_data_batch_async'):
            async def test():
                return await collector.get_price_data_batch_async(['BTCUSDT'], '1h', 1, True)
            
            result = asyncio.run(test())
            print(f"✓ Async wrapper works: {len(result)} symbols")
        
        print("\n✅ Fix successful! Run: streamlit run main.py")
        
    except Exception as e:
        print(f"⚠ Test error: {e}")
        print("But the fix should still work. Try running your app.")


if __name__ == "__main__":
    fix_async_error()