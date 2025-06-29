"""
Create __init__.py files for proper Python imports
Run this to ensure all infrastructure modules can be imported
Place in: crypto-dashboard-modular/create_init_files.py
"""

from pathlib import Path

def create_init_files():
    """Create __init__.py files in infrastructure directories"""
    
    # Define directories that need __init__.py
    dirs = [
        "infrastructure",
        "infrastructure/database", 
        "infrastructure/collectors",
        "infrastructure/message_bus",
        "infrastructure/scheduler"
    ]
    
    created = 0
    
    for dir_path in dirs:
        path = Path(dir_path)
        
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py
        init_file = path / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Package initialization"""\n')
            print(f"✅ Created {init_file}")
            created += 1
        else:
            print(f"   Already exists: {init_file}")
    
    print(f"\n✅ Created {created} __init__.py files")
    print("All infrastructure modules can now be imported properly")


if __name__ == "__main__":
    create_init_files()