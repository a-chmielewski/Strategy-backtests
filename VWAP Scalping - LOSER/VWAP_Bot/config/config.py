import yaml
from pathlib import Path
from threading import Lock

class Config:
    _instance = None
    _lock = Lock()
    
    def __new__(cls, config_path: str = 'VWAP_Bot\config\config.yaml'):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Config, cls).__new__(cls)
                    with open(config_path, 'r') as file:
                        cls._instance.config = yaml.safe_load(file)
        return cls._instance
    
    def get(self, *keys, default=None):
        data = self.config
        for key in keys:
            data = data.get(key, {})
        return data or default
