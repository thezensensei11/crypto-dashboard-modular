"""
Service Manager - Orchestrates all microservices
Place this file in: crypto-dashboard-modular/infrastructure/service_manager.py
"""

import asyncio
import signal
import sys
from typing import Dict, List, Optional
import logging
import subprocess
import psutil
import yaml
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import click

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ServiceConfig:
    """Configuration for a service"""
    name: str
    command: List[str]
    working_dir: str = "."
    env: Dict[str, str] = None
    depends_on: List[str] = None
    health_check: Optional[str] = None
    restart_policy: str = "on-failure"
    max_restarts: int = 3


class ServiceManager:
    """Manages all microservices"""
    
    def __init__(self, config_path: str = "infrastructure/services.yaml"):
        self.config_path = Path(config_path)
        self.services: Dict[str, ServiceConfig] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.statuses: Dict[str, ServiceStatus] = {}
        self._load_config()
        
    def _load_config(self):
        """Load service configuration"""
        if not self.config_path.exists():
            # Create default configuration
            self._create_default_config()
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        for name, service_config in config['services'].items():
            self.services[name] = ServiceConfig(
                name=name,
                command=service_config['command'],
                working_dir=service_config.get('working_dir', '.'),
                env=service_config.get('env', {}),
                depends_on=service_config.get('depends_on', []),
                health_check=service_config.get('health_check'),
                restart_policy=service_config.get('restart_policy', 'on-failure'),
                max_restarts=service_config.get('max_restarts', 3)
            )
            self.statuses[name] = ServiceStatus.STOPPED
    
    def _create_default_config(self):
        """Create default service configuration"""
        default_config = {
            'services': {
                'redis': {
                    'command': ['redis-server'],
                    'health_check': 'redis-cli ping'
                },
                'duckdb_api': {
                    'command': ['python', '-m', 'infrastructure.api.duckdb_api'],
                    'depends_on': ['redis'],
                    'env': {
                        'DB_PATH': 'crypto_data.duckdb',
                        'PORT': '8001'
                    }
                },
                'rest_collector': {
                    'command': ['python', '-m', 'infrastructure.collectors.collectors'],
                    'depends_on': ['redis', 'duckdb_api'],
                    'env': {
                        'COLLECTOR_TYPE': 'rest'
                    }
                },
                'websocket_collector': {
                    'command': ['python', '-m', 'infrastructure.collectors.collectors'],
                    'depends_on': ['redis'],
                    'env': {
                        'COLLECTOR_TYPE': 'websocket'
                    }
                },
                'data_processor': {
                    'command': ['python', '-m', 'infrastructure.processors.data_processor'],
                    'depends_on': ['redis', 'duckdb_api']
                },
                'scheduler': {
                    'command': ['celery', '-A', 'infrastructure.scheduler.scheduler', 'worker', '--beat', '--loglevel=info'],
                    'depends_on': ['redis'],
                    'env': {
                        'CELERY_BROKER': 'redis://localhost:6379/0'
                    }
                },
                'dashboard': {
                    'command': ['streamlit', 'run', 'main.py'],
                    'depends_on': ['duckdb_api'],
                    'env': {
                        'USE_DUCKDB': 'true',
                        'STREAMLIT_SERVER_PORT': '8501'
                    }
                }
            }
        }
        
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    async def start_service(self, name: str) -> bool:
        """Start a single service"""
        if name not in self.services:
            logger.error(f"Service {name} not found")
            return False
        
        if self.statuses[name] == ServiceStatus.RUNNING:
            logger.info(f"Service {name} is already running")
            return True
        
        service = self.services[name]
        
        # Check dependencies
        for dep in service.depends_on or []:
            if self.statuses.get(dep) != ServiceStatus.RUNNING:
                logger.info(f"Starting dependency {dep} for {name}")
                if not await self.start_service(dep):
                    logger.error(f"Failed to start dependency {dep}")
                    return False
        
        # Start the service
        self.statuses[name] = ServiceStatus.STARTING
        logger.info(f"Starting service {name}...")
        
        try:
            # Prepare environment
            env = {**os.environ, **(service.env or {})}
            
            # Start process
            process = subprocess.Popen(
                service.command,
                cwd=service.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.processes[name] = process
            
            # Wait a bit to check if it started successfully
            await asyncio.sleep(2)
            
            if process.poll() is None:
                self.statuses[name] = ServiceStatus.RUNNING
                logger.info(f"Service {name} started successfully (PID: {process.pid})")
                return True
            else:
                self.statuses[name] = ServiceStatus.ERROR
                stderr = process.stderr.read().decode() if process.stderr else ""
                logger.error(f"Service {name} failed to start: {stderr}")
                return False
                
        except Exception as e:
            self.statuses[name] = ServiceStatus.ERROR
            logger.error(f"Error starting service {name}: {e}")
            return False
    
    async def stop_service(self, name: str) -> bool:
        """Stop a single service"""
        if name not in self.services:
            logger.error(f"Service {name} not found")
            return False
        
        if self.statuses[name] != ServiceStatus.RUNNING:
            logger.info(f"Service {name} is not running")
            return True
        
        self.statuses[name] = ServiceStatus.STOPPING
        logger.info(f"Stopping service {name}...")
        
        process = self.processes.get(name)
        if process:
            # Try graceful shutdown
            process.terminate()
            
            # Wait for process to stop
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                logger.warning(f"Force killing service {name}")
                process.kill()
                process.wait()
            
            del self.processes[name]
        
        self.statuses[name] = ServiceStatus.STOPPED
        logger.info(f"Service {name} stopped")
        return True
    
    async def restart_service(self, name: str) -> bool:
        """Restart a service"""
        await self.stop_service(name)
        await asyncio.sleep(1)
        return await self.start_service(name)
    
    async def start_all(self):
        """Start all services"""
        # Sort services by dependencies
        sorted_services = self._topological_sort()
        
        for name in sorted_services:
            await self.start_service(name)
    
    async def stop_all(self):
        """Stop all services"""
        # Stop in reverse order
        sorted_services = self._topological_sort()
        
        for name in reversed(sorted_services):
            await self.stop_service(name)
    
    def _topological_sort(self) -> List[str]:
        """Sort services by dependencies"""
        visited = set()
        result = []
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            
            service = self.services.get(name)
            if service:
                for dep in service.depends_on or []:
                    visit(dep)
                result.append(name)
        
        for name in self.services:
            visit(name)
        
        return result
    
    async def monitor_services(self):
        """Monitor services and restart if needed"""
        while True:
            for name, process in list(self.processes.items()):
                if process.poll() is not None:
                    # Process has exited
                    logger.warning(f"Service {name} has exited")
                    self.statuses[name] = ServiceStatus.ERROR
                    
                    service = self.services[name]
                    if service.restart_policy == 'always' or service.restart_policy == 'on-failure':
                        logger.info(f"Restarting service {name}")
                        await self.start_service(name)
            
            await asyncio.sleep(5)
    
    def get_status(self) -> Dict[str, Dict]:
        """Get status of all services"""
        status = {}
        
        for name, service in self.services.items():
            process = self.processes.get(name)
            
            status[name] = {
                'status': self.statuses[name].value,
                'pid': process.pid if process and process.poll() is None else None,
                'command': ' '.join(service.command),
                'depends_on': service.depends_on or []
            }
        
        return status


# CLI Interface
@click.group()
def cli():
    """Crypto Dashboard Service Manager"""
    pass


@cli.command()
@click.option('--service', '-s', help='Service name to start (or "all")')
def start(service):
    """Start services"""
    manager = ServiceManager()
    
    async def run():
        if service == 'all':
            await manager.start_all()
        else:
            await manager.start_service(service)
        
        # Keep running and monitor
        await manager.monitor_services()
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nShutting down...")
        asyncio.run(manager.stop_all())


@cli.command()
@click.option('--service', '-s', help='Service name to stop (or "all")')
def stop(service):
    """Stop services"""
    manager = ServiceManager()
    
    async def run():
        if service == 'all':
            await manager.stop_all()
        else:
            await manager.stop_service(service)
    
    asyncio.run(run())


@cli.command()
def status():
    """Show service status"""
    manager = ServiceManager()
    status = manager.get_status()
    
    # Print status table
    print("\nService Status:")
    print("-" * 60)
    print(f"{'Service':<20} {'Status':<15} {'PID':<10} {'Depends On'}")
    print("-" * 60)
    
    for name, info in status.items():
        deps = ', '.join(info['depends_on']) if info['depends_on'] else 'None'
        pid = str(info['pid']) if info['pid'] else '-'
        print(f"{name:<20} {info['status']:<15} {pid:<10} {deps}")


@cli.command()
@click.option('--service', '-s', required=True, help='Service name to restart')
def restart(service):
    """Restart a service"""
    manager = ServiceManager()
    
    async def run():
        await manager.restart_service(service)
    
    asyncio.run(run())


# Docker Compose Generator
@cli.command()
def generate_compose():
    """Generate docker-compose.yml"""
    compose = {
        'version': '3.8',
        'services': {
            'redis': {
                'image': 'redis:7-alpine',
                'ports': ['6379:6379'],
                'volumes': ['redis_data:/data']
            },
            'rest_collector': {
                'build': '.',
                'command': 'python -m infrastructure.collectors.collectors',
                'environment': {
                    'COLLECTOR_TYPE': 'rest',
                    'REDIS_URL': 'redis://redis:6379'
                },
                'depends_on': ['redis']
            },
            'websocket_collector': {
                'build': '.',
                'command': 'python -m infrastructure.collectors.collectors',
                'environment': {
                    'COLLECTOR_TYPE': 'websocket',
                    'REDIS_URL': 'redis://redis:6379'
                },
                'depends_on': ['redis']
            },
            'data_processor': {
                'build': '.',
                'command': 'python -m infrastructure.processors.data_processor',
                'environment': {
                    'REDIS_URL': 'redis://redis:6379',
                    'DB_PATH': '/data/crypto_data.duckdb'
                },
                'volumes': ['./data:/data'],
                'depends_on': ['redis']
            },
            'scheduler': {
                'build': '.',
                'command': 'celery -A infrastructure.scheduler.scheduler worker --beat --loglevel=info',
                'environment': {
                    'CELERY_BROKER': 'redis://redis:6379/0',
                    'DB_PATH': '/data/crypto_data.duckdb'
                },
                'volumes': ['./data:/data'],
                'depends_on': ['redis']
            },
            'dashboard': {
                'build': '.',
                'command': 'streamlit run main.py',
                'ports': ['8501:8501'],
                'environment': {
                    'USE_DUCKDB': 'true',
                    'DB_PATH': '/data/crypto_data.duckdb'
                },
                'volumes': ['./data:/data'],
                'depends_on': ['redis']
            }
        },
        'volumes': {
            'redis_data': {},
            'crypto_data': {}
        }
    }
    
    with open('docker-compose.yml', 'w') as f:
        yaml.dump(compose, f, default_flow_style=False)
    
    print("Generated docker-compose.yml")


# Main entry point
if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    cli()