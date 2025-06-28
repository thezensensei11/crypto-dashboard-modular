"""
Settings management for the crypto dashboard
Handles configuration loading and saving
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Settings:
    """Centralized settings management"""
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path(".")
        self.config_dir.mkdir(exist_ok=True)
        
        # File paths
        self.universe_file = self.config_dir / "universe_config.json"
        self.columns_file = self.config_dir / "columns_config.json"
        self.state_file = self.config_dir / "dashboard_state.json"
        self.metrics_file = self.config_dir / "metrics_data.parquet"
        self.config_file = self.config_dir / "config.yaml"
        
        # Load configurations
        self._load_all()
    
    def _load_all(self):
        """Load all configuration files"""
        self.universe = self._load_json(self.universe_file, default=[])
        self.columns_config = self._load_json(self.columns_file, default=[])
        self.dashboard_state = self._load_json(self.state_file, default={})
        self.app_config = self._load_yaml(self.config_file, default={
            'data': {
                'primary_source': 'binance_futures',
                'cache': {'enabled': True}
            },
            'metrics': {
                'benchmark': 'BTCUSDT',
                'default_lookback': 30
            },
            'ui': {
                'theme': 'dark',
                'refresh_interval': 5
            }
        })
    
    def _load_json(self, path: Path, default: Any = None) -> Any:
        """Load JSON file with error handling"""
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
        return default
    
    def _load_yaml(self, path: Path, default: Any = None) -> Any:
        """Load YAML file with error handling"""
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
        return default
    
    def save_universe(self, universe: List[str]):
        """Save universe configuration"""
        self.universe = universe
        self._save_json(self.universe_file, universe)
        self._update_state('universe', universe)
    
    def save_columns(self, columns: List[Dict]):
        """Save columns configuration"""
        self.columns_config = columns
        self._save_json(self.columns_file, columns)
        self._update_state('columns_config', columns)
    
    def save_state(self, **kwargs):
        """Save dashboard state with additional fields"""
        self.dashboard_state.update({
            'universe': self.universe,
            'columns_config': self.columns_config,
            'last_refresh': kwargs.get('last_refresh'),
            'saved_at': datetime.now().isoformat()
        })
        self.dashboard_state.update(kwargs)
        self._save_json(self.state_file, self.dashboard_state)
    
    def _save_json(self, path: Path, data: Any):
        """Save data to JSON file"""
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving {path}: {e}")
    
    def _update_state(self, key: str, value: Any):
        """Update a specific key in the state file"""
        self.dashboard_state[key] = value
        self.save_state()
    
    def get_last_refresh(self) -> Optional[datetime]:
        """Get last refresh timestamp"""
        if 'last_refresh' in self.dashboard_state and self.dashboard_state['last_refresh']:
            try:
                return datetime.fromisoformat(self.dashboard_state['last_refresh'])
            except:
                pass
        return None
    
    def reset_all(self):
        """Reset all configurations"""
        self.universe = []
        self.columns_config = []
        self.dashboard_state = {}
        
        # Delete files
        for file in [self.universe_file, self.columns_file, self.state_file, self.metrics_file]:
            if file.exists():
                file.unlink()
                logger.info(f"Deleted {file}")

# Singleton instance
_settings_instance: Optional[Settings] = None

def get_settings() -> Settings:
    """Get or create settings instance"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
