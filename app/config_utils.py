"""
Configuration utilities for the chatbot project
"""
import toml
import os
from typing import Dict, Any


def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """
    Load configuration from TOML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return toml.load(config_path)


def get_api_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get API configuration settings.

    Args:
        config: Configuration dictionary (if None, loads from default path)

    Returns:
        Dictionary containing API configuration
    """
    if config is None:
        config = load_config()

    return config.get("api", {
        "base_url": "http://localhost:8001",
        "port": 8001,
        "host": "0.0.0.0"
    })


def get_quality_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get quality configuration settings.

    Args:
        config: Configuration dictionary (if None, loads from default path)

    Returns:
        Dictionary containing quality configuration
    """
    if config is None:
        config = load_config()

    return config.get("quality", {
        "minimum_overall_score": 1.0,
        "minimum_critical_aspects_score": 1.0,
        "excellent_threshold": 4.0,
        "good_threshold": 3.0
    })
