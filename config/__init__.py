"""Configuration package for LLM Tabular Preprocessing.

Loads YAML configuration files for models, prompts, and logging.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

CONFIG_DIR = Path(__file__).parent

def load_yaml(filename: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        filename: Name of YAML file (e.g., 'model_config.yaml')
    
    Returns:
        Dictionary with configuration
    """
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def get_model_config() -> Dict[str, Any]:
    """Load model configuration"""
    return load_yaml('model_config.yaml')

def get_prompt_templates() -> Dict[str, Any]:
    """Load prompt templates"""
    return load_yaml('prompt_templates.yaml')

def get_logging_config() -> Dict[str, Any]:
    """Load logging configuration"""
    return load_yaml('logging_config.yaml')

__all__ = ['load_yaml', 'get_model_config', 'get_prompt_templates', 'get_logging_config']
