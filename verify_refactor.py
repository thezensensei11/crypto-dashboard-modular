#!/usr/bin/env python3
"""
Verification script to check if refactoring was successful
Run after you've pasted all the code: python verify_refactor.py
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import sys
from pathlib import Path
import importlib.util
import subprocess

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


class RefactorVerifier:
    def __init__(self):
        self.root = Path.cwd()
        self.errors = []
        self.warnings = []
        self.success = []
        
    def print_header(self, text):
        print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
        print(f"{BOLD}{BLUE}{text.center(60)}{RESET}")
        print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")
        
    def check_file_exists(self, path, required=True):
        """Check if a file exists"""
        full_path = self.root / path
        exists = full_path.exists()
        
        if exists:
            # Check if file has actual content (not just TODO)
            with open(full_path, 'r') as f:
                content = f.read()
                if "# TODO:" in content and len(content) < 200:
                    self.warnings.append(f"File exists but appears empty: {path}")
                    print(f"{YELLOW}⚠ {path} - exists but needs code{RESET}")
                    return False
                else:
                    self.success.append(f"File ready: {path}")
                    print(f"{GREEN}✓ {path}{RESET}")
                    return True
        else:
            if required:
                self.errors.append(f"Missing required file: {path}")
                print(f"{RED}✗ {path} - MISSING{RESET}")
            else:
                self.warnings.append(f"Optional file missing: {path}")
                print(f"{YELLOW}⚠ {path} - optional, not found{RESET}")
            return False
    
    def check_python_syntax(self, path):
        """Check if Python file has valid syntax"""
        full_path = self.root / path
        if not full_path.exists():
            return False
            
        try:
            with open(full_path, 'r') as f:
                compile(f.read(), path, 'exec')
            return True
        except SyntaxError as e:
            self.errors.append(f"Syntax error in {path}: {e}")
            print(f"{RED}  Syntax error on line {e.lineno}{RESET}")
            return False
        except Exception as e:
            self.warnings.append(f"Could not check {path}: {e}")
            return False
    
    def check_imports(self, path, imports):
        """Check if file can import required modules"""
        full_path = self.root / path
        if not full_path.exists():
            return False
            
        for imp in imports:
            try:
                spec = importlib.util.spec_from_file_location("test_module", full_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if not hasattr(module, imp):
                        self.warnings.append(f"{path} missing: {imp}")
                        print(f"{YELLOW}  Missing: {imp}{RESET}")
            except Exception as e:
                self.warnings.append(f"Import check failed for {path}: {e}")
                return False
        return True
    
    def verify_core_files(self):
        """Verify core module files"""
        self.print_header("CHECKING CORE MODULE")
        
        core_files = {
            "core/config.py": ["Settings", "get_settings"],
            "core/models.py": ["OHLCVData", "Event", "EventType"],
            "core/constants.py": [],
            "core/exceptions.py": ["CryptoDataException"],
            "core/__init__.py": []
        }
        
        all_good = True
        for file, expected_items in core_files.items():
            if self.check_file_exists(file):
                if file.endswith('.py') and file != "core/__init__.py":
                    if not self.check_python_syntax(file):
                        all_good = False
                    elif expected_items:
                        self.check_imports(file, expected_items)
            else:
                all_good = False
        
        return all_good
    
    def verify_infrastructure_files(self):
        """Verify infrastructure files"""
        self.print_header("CHECKING INFRASTRUCTURE")
        
        infrastructure_files = {
            # Message Bus
            "infrastructure/message_bus/bus.py": ["MessageBus"],
            "infrastructure/message_bus/__init__.py": [],
            
            # Database
            "infrastructure/database/manager.py": ["DuckDBManager"],
            "infrastructure/database/__init__.py": [],
            
            # Collectors
            "infrastructure/collectors/base.py": ["BaseCollector"],
            "infrastructure/collectors/websocket.py": ["WebSocketCollector"],
            "infrastructure/collectors/rest.py": ["RESTCollector"],
            "infrastructure/collectors/__init__.py": [],
            
            # Processors
            "infrastructure/processors/data_processor.py": ["DataProcessor"],
            "infrastructure/processors/__init__.py": [],
            
            # Scheduler
            "infrastructure/scheduler/celery_app.py": ["app"],
            "infrastructure/scheduler/tasks.py": [],
            "infrastructure/scheduler/__init__.py": [],
            
            # Root init
            "infrastructure/__init__.py": []
        }
        
        all_good = True
        for file, expected_items in infrastructure_files.items():
            if self.check_file_exists(file):
                if file.endswith('.py') and '__init__' not in file:
                    if not self.check_python_syntax(file):
                        all_good = False
            else:
                all_good = False
        
        return all_good
    
    def verify_scripts(self):
        """Verify script files"""
        self.print_header("CHECKING SCRIPTS")
        
        scripts = [
            "scripts/init_infrastructure.py",
            "scripts/run_collector.py",
            "scripts/run_processor.py",
            "scripts/test_infrastructure.py",
            "scripts/migrate_data.py",
            "scripts/manual_backfill.py",
            "scripts/monitor.sh",
            "scripts/__init__.py"
        ]
        
        all_good = True
        for script in scripts:
            if self.check_file_exists(script):
                if script.endswith('.py') and script != "scripts/__init__.py":
                    if not self.check_python_syntax(script):
                        all_good = False
                elif script.endswith('.sh'):
                    # Check if shell script is executable
                    full_path = self.root / script
                    if not os.access(full_path, os.X_OK):
                        self.warnings.append(f"{script} not executable")
                        print(f"{YELLOW}  Not executable (run: chmod +x {script}){RESET}")
            else:
                all_good = False
        
        return all_good
    
    def verify_root_files(self):
        """Verify root configuration files"""
        self.print_header("CHECKING ROOT FILES")
        
        root_files = [
            ("docker-compose.yml", True),
            ("Dockerfile", True),
            (".env.example", True),
            ("setup.sh", True),
            ("main.py", True),
            ("requirements.txt", True),
            (".env", False),  # Optional
            ("README_NEW.md", True),
            ("data/infrastructure_adapter.py", True)
        ]
        
        all_good = True
        for file, required in root_files:
            if not self.check_file_exists(file, required):
                if required:
                    all_good = False
        
        # Check if setup.sh is executable
        setup_sh = self.root / "setup.sh"
        if setup_sh.exists() and not os.access(setup_sh, os.X_OK):
            self.warnings.append("setup.sh not executable")
            print(f"{YELLOW}  setup.sh not executable (run: chmod +x setup.sh){RESET}")
        
        return all_good
    
    def check_dependencies(self):
        """Check if required dependencies are in requirements.txt"""
        self.print_header("CHECKING DEPENDENCIES")
        
        req_file = self.root / "requirements.txt"
        if not req_file.exists():
            self.errors.append("requirements.txt not found")
            return False
        
        with open(req_file, 'r') as f:
            requirements = f.read().lower()
        
        required_packages = [
            "duckdb", "redis", "celery", "websockets", 
            "pydantic", "flower", "aioredis"
        ]
        
        missing = []
        for pkg in required_packages:
            if pkg not in requirements:
                missing.append(pkg)
        
        if missing:
            self.errors.append(f"Missing dependencies: {', '.join(missing)}")
            print(f"{RED}✗ Missing dependencies: {', '.join(missing)}{RESET}")
            return False
        else:
            self.success.append("All required dependencies present")
            print(f"{GREEN}✓ All required dependencies found{RESET}")
            return True
    
    def check_old_files_removed(self):
        """Check if old files were properly removed"""
        self.print_header("CHECKING OLD FILES REMOVED")
        
        old_files = [
            "binance_data_collector.py",
            "smart_data_manager.py",
            "price_metrics.py",
            "debug.py",
            "test_imports.py",
            "data/binance_data_collector.py",
            "data/duckdb_collector.py",
            "data/smart_data_manager.py"
        ]
        
        found_old = []
        for old_file in old_files:
            if (self.root / old_file).exists():
                found_old.append(old_file)
        
        if found_old:
            self.warnings.append(f"Old files still present: {', '.join(found_old)}")
            for f in found_old:
                print(f"{YELLOW}⚠ Old file still exists: {f}{RESET}")
            return False
        else:
            self.success.append("All old files removed")
            print(f"{GREEN}✓ All old files have been removed{RESET}")
            return True
    
    def check_services(self):
        """Check if services can be started"""
        self.print_header("CHECKING SERVICES")
        
        # Check Redis
        try:
            result = subprocess.run(['redis-cli', 'ping'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and 'PONG' in result.stdout:
                self.success.append("Redis is running")
                print(f"{GREEN}✓ Redis is running{RESET}")
            else:
                self.warnings.append("Redis not running")
                print(f"{YELLOW}⚠ Redis not running (start with: redis-server){RESET}")
        except:
            self.warnings.append("Redis not installed")
            print(f"{YELLOW}⚠ Redis not found (install redis){RESET}")
        
        # Check if Docker is available
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                self.success.append("Docker is available")
                print(f"{GREEN}✓ Docker is available{RESET}")
            else:
                self.warnings.append("Docker not available")
                print(f"{YELLOW}⚠ Docker not available{RESET}")
        except:
            self.warnings.append("Docker not installed")
            print(f"{YELLOW}⚠ Docker not found{RESET}")
    
    def generate_report(self):
        """Generate final verification report"""
        self.print_header("VERIFICATION REPORT")
        
        total_checks = len(self.success) + len(self.warnings) + len(self.errors)
        
        print(f"{BOLD}Summary:{RESET}")
        print(f"{GREEN}✓ Passed: {len(self.success)}{RESET}")
        print(f"{YELLOW}⚠ Warnings: {len(self.warnings)}{RESET}")
        print(f"{RED}✗ Errors: {len(self.errors)}{RESET}")
        
        if self.errors:
            print(f"\n{BOLD}{RED}Errors to fix:{RESET}")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n{BOLD}{YELLOW}Warnings to review:{RESET}")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        # Next steps
        print(f"\n{BOLD}Next Steps:{RESET}")
        
        if self.errors:
            print("1. Fix the errors listed above")
            print("2. Make sure all files have actual code (not just TODOs)")
            print("3. Run this verification again")
        elif self.warnings and "needs code" in ' '.join(self.warnings):
            print("1. Paste the code into files that show 'needs code'")
            print("2. Make scripts executable: chmod +x setup.sh scripts/monitor.sh")
            print("3. Run this verification again")
        else:
            print("1. Make scripts executable: chmod +x setup.sh scripts/monitor.sh")
            print("2. Copy .env.example to .env and configure")
            print("3. Run: ./setup.sh")
            print("4. Test: python scripts/test_infrastructure.py")
            print("5. Start services: docker-compose up -d")
        
        # Save report
        report_file = self.root / "VERIFICATION_REPORT.txt"
        with open(report_file, 'w') as f:
            f.write("Refactoring Verification Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Passed: {len(self.success)}\n")
            f.write(f"Warnings: {len(self.warnings)}\n")
            f.write(f"Errors: {len(self.errors)}\n\n")
            
            if self.errors:
                f.write("ERRORS:\n")
                for error in self.errors:
                    f.write(f"- {error}\n")
                f.write("\n")
            
            if self.warnings:
                f.write("WARNINGS:\n")
                for warning in self.warnings:
                    f.write(f"- {warning}\n")
        
        print(f"\n{BLUE}Report saved to: VERIFICATION_REPORT.txt{RESET}")
    
    def run(self):
        """Run all verification checks"""
        self.print_header("CRYPTO DASHBOARD REFACTORING VERIFICATION")
        
        # Run all checks
        self.verify_core_files()
        self.verify_infrastructure_files()
        self.verify_scripts()
        self.verify_root_files()
        self.check_dependencies()
        self.check_old_files_removed()
        self.check_services()
        
        # Generate report
        self.generate_report()


if __name__ == "__main__":
    verifier = RefactorVerifier()
    verifier.run()