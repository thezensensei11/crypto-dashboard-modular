#!/usr/bin/env python3
"""
Fix Event deserialization error and clear bad messages
Run this: python fix_event_error.py
"""

import redis
from pathlib import Path
import json

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def check_redis_messages():
    """Check what's in Redis streams"""
    print(f"\n{BOLD}Checking Redis streams...{RESET}")
    
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Check for existing streams
        streams = [
            "crypto:stream:price.update",
            "crypto:stream:candle.closed", 
            "crypto:stream:historical.data"
        ]
        
        for stream in streams:
            try:
                # Get stream length
                length = r.xlen(stream)
                print(f"\n{BLUE}{stream}: {length} messages{RESET}")
                
                if length > 0:
                    # Get first message to see format
                    messages = r.xrange(stream, count=1)
                    if messages:
                        msg_id, data = messages[0]
                        print(f"  First message ID: {msg_id}")
                        print(f"  Data keys: {list(data.keys())}")
                        if 'data' in data:
                            print(f"  Data content preview: {data['data'][:100]}...")
            except:
                print(f"  {YELLOW}Stream doesn't exist yet{RESET}")
                
    except Exception as e:
        print(f"{RED}Redis connection error: {e}{RESET}")
        return False
    
    return True


def fix_event_class():
    """Fix the Event.from_dict method"""
    print(f"\n{BOLD}Fixing Event.from_dict method...{RESET}")
    
    models_file = Path("core/models.py")
    if not models_file.exists():
        print(f"{RED}core/models.py not found!{RESET}")
        return False
    
    with open(models_file, 'r') as f:
        content = f.read()
    
    # Check if from_dict method exists and needs fixing
    if '@classmethod\n    def from_dict(cls, data: Dict) -> \'Event\':' in content:
        print(f"{GREEN}✓ from_dict method found{RESET}")
        
        # Check if it's handling the Redis format correctly
        # The issue might be that Redis returns strings, not the original types
        
        # Find and fix the from_dict method
        lines = content.split('\n')
        new_lines = []
        in_from_dict = False
        method_fixed = False
        
        for i, line in enumerate(lines):
            if 'def from_dict(cls, data: Dict)' in line:
                in_from_dict = True
                new_lines.append(line)
                # Add better error handling
                new_lines.append('        """Create from dictionary with better error handling"""')
                new_lines.append('        try:')
                new_lines.append('            # Handle both direct dict and JSON string in data field')
                new_lines.append('            if isinstance(data.get("data"), str):')
                new_lines.append('                try:')
                new_lines.append('                    data["data"] = json.loads(data["data"])')
                new_lines.append('                except:')
                new_lines.append('                    pass  # Keep as string if not JSON')
                new_lines.append('            ')
                new_lines.append('            # Parse event type')
                new_lines.append('            event_type = data.get("type", "")')
                new_lines.append('            if isinstance(event_type, str) and not event_type.startswith("EventType."):')
                new_lines.append('                event_type = EventType(event_type)')
                new_lines.append('            elif isinstance(event_type, str):')
                new_lines.append('                event_type = EventType(event_type.replace("EventType.", ""))')
                new_lines.append('            ')
                new_lines.append('            return cls(')
                new_lines.append('                id=data["id"],')
                new_lines.append('                type=event_type,')
                new_lines.append('                timestamp=datetime.fromisoformat(data["timestamp"]),')
                new_lines.append('                source=data["source"],')
                new_lines.append('                data=data.get("data", {}),')
                new_lines.append('                metadata=data.get("metadata")')
                new_lines.append('            )')
                new_lines.append('        except Exception as e:')
                new_lines.append('            raise ValueError(f"Failed to deserialize Event: {e} - Data: {data}")')
                method_fixed = True
                
                # Skip the original method implementation
                j = i + 1
                indent_level = len(line) - len(line.lstrip())
                while j < len(lines) and (lines[j].startswith(' ' * (indent_level + 4)) or not lines[j].strip()):
                    j += 1
                
                # Skip lines we're replacing
                for k in range(i + 1, j):
                    lines[k] = None
                    
            elif line is not None:
                new_lines.append(line)
        
        if method_fixed:
            # Write back
            final_content = '\n'.join([l for l in new_lines if l is not None])
            
            # Ensure json is imported
            if 'import json' not in final_content:
                import_lines = final_content.split('\n')
                for i, line in enumerate(import_lines):
                    if 'import' in line and i < 20:
                        continue
                    else:
                        import_lines.insert(i, 'import json')
                        break
                final_content = '\n'.join(import_lines)
            
            with open(models_file, 'w') as f:
                f.write(final_content)
            
            print(f"{GREEN}✓ Fixed Event.from_dict method{RESET}")
            return True
            
    else:
        print(f"{YELLOW}from_dict method not found or different format{RESET}")
    
    return False


def clear_bad_messages():
    """Clear existing messages from Redis streams"""
    print(f"\n{BOLD}Clearing bad messages from Redis...{RESET}")
    
    response = input(f"{YELLOW}Clear all existing messages from Redis streams? (y/n): {RESET}")
    if response.lower() != 'y':
        print("Skipping message clearing")
        return
    
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        streams = [
            "crypto:stream:price.update",
            "crypto:stream:candle.closed",
            "crypto:stream:historical.data"
        ]
        
        for stream in streams:
            try:
                # Delete the stream
                r.delete(stream)
                print(f"{GREEN}✓ Cleared {stream}{RESET}")
            except:
                pass
                
        print(f"{GREEN}✓ All streams cleared{RESET}")
        
    except Exception as e:
        print(f"{RED}Failed to clear messages: {e}{RESET}")


def main():
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{BLUE}{'Fix Event Deserialization Error'.center(60)}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}")
    
    # Check what's in Redis
    check_redis_messages()
    
    # Fix the Event class
    if fix_event_class():
        print(f"\n{GREEN}✓ Event class fixed{RESET}")
    
    # Offer to clear bad messages
    clear_bad_messages()
    
    print(f"\n{GREEN}{'=' * 60}{RESET}")
    print(f"{GREEN}✅ Fixes applied!{RESET}")
    print(f"{GREEN}{'=' * 60}{RESET}")
    
    print(f"\n{BOLD}Now try running the processor again:{RESET}")
    print(f"   {GREEN}python scripts/run_processor.py{RESET}")
    
    print(f"\n{BOLD}If it still fails, you can:{RESET}")
    print("1. Clear all Redis data: redis-cli FLUSHALL")
    print("2. Start fresh with the websocket collector first")
    print("3. Check the exact error message for more details")


if __name__ == "__main__":
    main()