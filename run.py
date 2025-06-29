#!/usr/bin/env python3
"""
Helper to run scripts from project root
Usage: python run.py <script_path> [args...]
"""

import sys
import subprocess
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python run.py <script_path> [args...]")
    print("Example: python run.py scripts/test_infrastructure.py")
    sys.exit(1)

script_path = sys.argv[1]
args = sys.argv[2:]

# Run from project root
project_root = Path(__file__).parent
cmd = [sys.executable, script_path] + args

print(f"Running from {project_root}: {' '.join(cmd)}")
subprocess.run(cmd, cwd=project_root)
