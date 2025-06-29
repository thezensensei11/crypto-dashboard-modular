#!/usr/bin/env python3
"""
Check which files still have TODO comments and need code to be pasted
Run this to track your progress: python check_todos.py
"""

import os
from pathlib import Path
import re

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class TodoChecker:
    def __init__(self):
        self.root = Path.cwd()
        self.files_with_todos = []
        self.files_completed = []
        self.files_to_check = [
            # Core files
            "core/config.py",
            "core/models.py",
            "core/constants.py",
            "core/exceptions.py",
            
            # Infrastructure files
            "infrastructure/message_bus/bus.py",
            "infrastructure/database/manager.py",
            "infrastructure/collectors/base.py",
            "infrastructure/collectors/websocket.py",
            "infrastructure/collectors/rest.py",
            "infrastructure/processors/data_processor.py",
            "infrastructure/scheduler/celery_app.py",
            "infrastructure/scheduler/tasks.py",
            
            # Scripts
            "scripts/init_infrastructure.py",
            "scripts/run_collector.py",
            "scripts/run_processor.py",
            "scripts/test_infrastructure.py",
            "scripts/migrate_data.py",
            "scripts/manual_backfill.py",
            "scripts/monitor.sh",
            
            # Other files
            "data/infrastructure_adapter.py",
            "docker-compose.yml",
            "Dockerfile",
            ".env.example",
            "setup.sh",
            "main.py",
        ]
    
    def check_file(self, file_path: str) -> dict:
        """Check if file has TODO or actual code"""
        path = self.root / file_path
        
        if not path.exists():
            return {"status": "missing", "size": 0, "has_todo": False}
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check for TODO comments
        has_todo = "# TODO:" in content or "TODO:" in content
        
        # Check file size (files with just TODO are usually < 200 bytes)
        size = len(content)
        
        # Determine status
        if has_todo and size < 500:
            status = "needs_code"
        elif size < 100 and not file_path.endswith(('__init__.py', '.env.example')):
            status = "possibly_empty"
        else:
            status = "has_code"
        
        return {
            "status": status,
            "size": size,
            "has_todo": has_todo,
            "artifact_hint": self.extract_artifact_hint(content) if has_todo else None
        }
    
    def extract_artifact_hint(self, content: str) -> str:
        """Extract which artifact to copy from"""
        match = re.search(r"Copy from '([^']+)' artifact", content)
        if match:
            return match.group(1)
        return "Check file for instructions"
    
    def run_check(self):
        """Check all files"""
        print(f"\n{BOLD}{BLUE}{'=' * 70}{RESET}")
        print(f"{BOLD}{BLUE}{'TODO File Checker - Track Your Progress'.center(70)}{RESET}")
        print(f"{BOLD}{BLUE}{'=' * 70}{RESET}\n")
        
        needs_code = []
        has_code = []
        missing = []
        
        for file_path in self.files_to_check:
            result = self.check_file(file_path)
            
            if result["status"] == "missing":
                missing.append(file_path)
                print(f"{RED}✗ {file_path:<50} - MISSING{RESET}")
            elif result["status"] == "needs_code":
                needs_code.append((file_path, result["artifact_hint"]))
                print(f"{YELLOW}⚠ {file_path:<50} - NEEDS CODE{RESET}")
                if result["artifact_hint"]:
                    print(f"  {BLUE}→ Copy from: {result['artifact_hint']}{RESET}")
            else:
                has_code.append(file_path)
                print(f"{GREEN}✓ {file_path:<50} - HAS CODE{RESET}")
        
        # Summary
        print(f"\n{BOLD}{'=' * 70}{RESET}")
        print(f"{BOLD}Progress Summary:{RESET}")
        print(f"{GREEN}✓ Completed: {len(has_code)}/{len(self.files_to_check)} files{RESET}")
        print(f"{YELLOW}⚠ Need code: {len(needs_code)} files{RESET}")
        if missing:
            print(f"{RED}✗ Missing: {len(missing)} files{RESET}")
        
        # Detailed list of files needing code
        if needs_code:
            print(f"\n{BOLD}{YELLOW}Files that still need code:{RESET}")
            print(f"{YELLOW}{'=' * 70}{RESET}")
            for i, (file_path, artifact) in enumerate(needs_code, 1):
                print(f"{YELLOW}{i:2d}. {file_path:<45} → {artifact}{RESET}")
        
        # Next steps
        if needs_code:
            print(f"\n{BOLD}Next Steps:{RESET}")
            print("1. Open each file listed above")
            print("2. Find the artifact mentioned (use Ctrl+F in the conversation)")
            print("3. Copy the specified code section")
            print("4. Replace ALL content in the file (including TODO comments)")
            print("5. Save the file")
            print("\nThen run this script again to check progress!")
        else:
            print(f"\n{GREEN}{BOLD}🎉 All files have code!{RESET}")
            print("\nNext steps:")
            print("1. Run: python update_init_files.py")
            print("2. Run: python verify_refactor.py")
            print("3. Make scripts executable: chmod +x setup.sh scripts/monitor.sh")
            print("4. Run: ./setup.sh")
        
        # Save progress report
        self.save_progress_report(has_code, needs_code, missing)
    
    def save_progress_report(self, has_code, needs_code, missing):
        """Save a progress report file"""
        with open("TODO_PROGRESS.txt", "w") as f:
            f.write("TODO Progress Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total files: {len(self.files_to_check)}\n")
            f.write(f"Completed: {len(has_code)}\n")
            f.write(f"Need code: {len(needs_code)}\n")
            f.write(f"Missing: {len(missing)}\n\n")
            
            if needs_code:
                f.write("FILES NEEDING CODE:\n")
                f.write("-" * 50 + "\n")
                for file_path, artifact in needs_code:
                    f.write(f"{file_path}\n")
                    f.write(f"  → Copy from: {artifact}\n\n")
        
        print(f"\n{BLUE}Progress report saved to: TODO_PROGRESS.txt{RESET}")


if __name__ == "__main__":
    checker = TodoChecker()
    checker.run_check()