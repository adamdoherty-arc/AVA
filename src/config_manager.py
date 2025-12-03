"""
Configuration Manager for Magnus Trading Dashboard

Centralized configuration management with:
- YAML file parsing
- Environment variable overrides
- Schema validation
- Hot reload support
- Singleton pattern
- Type hints and defaults
"""

import os
import yaml
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Singleton configuration manager for Magnus Trading Dashboard.

    Features:
    - Load configuration from YAML files
    - Override with environment variables (MAGNUS_SECTION_KEY pattern)
    - Schema validation
    - Thread-safe access
    - Hot reload support
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize configuration manager (only once)"""
        if self._initialized:
            return

        self._initialized = True
        self._config: Dict[str, Any] = {}
        self._config_dir = Path(__file__).parent.parent / "config"
        self._last_reload: Optional[datetime] = None
        self._file_mtimes: Dict[Path, float] = {}

        # Load all configuration files
        self._load_all_configs()

        logger.info(f"ConfigManager initialized with {len(self._config)} sections")

    def _load_all_configs(self) -> None:
        """Load all YAML configuration files"""
        config_files = {
            "default": self._config_dir / "default.yaml",
            "pages": self._config_dir / "pages.yaml",
            "features": self._config_dir / "features.yaml",
            "services": self._config_dir / "services.yaml",
        }

        for config_name, config_path in config_files.items():
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f) or {}

                    # Merge into main config
                    self._config.update(config_data)

                    # Track file modification time for hot reload
                    self._file_mtimes[config_path] = config_path.stat().st_mtime

                    logger.debug(f"Loaded {config_name} configuration from {config_path}")
                except Exception as e:
                    logger.error(f"Error loading {config_name} config: {e}")
            else:
                logger.warning(f"Config file not found: {config_path}")

        self._last_reload = datetime.now()

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides.
        Pattern: MAGNUS_SECTION_KEY overrides config[section][key]
        Example: MAGNUS_DATABASE_POOL_MAX=20 overrides database.pool_max
        """
        env_prefix = "MAGNUS_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(env_prefix):
                continue

            # Parse MAGNUS_SECTION_KEY format
            config_path = env_key[len(env_prefix):].lower().split('_')

            if len(config_path) < 2:
                continue

            # Navigate to the config location
            current = self._config
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value with type conversion
            final_key = config_path[-1]
            current[final_key] = self._convert_env_value(env_value)

            logger.debug(f"Applied env override: {env_key} = {env_value}")

    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type"""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # String
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation key.

        Args:
            key: Configuration key in dot notation (e.g., "database.pool_max")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config = ConfigManager()
            >>> config.get("database.pool_max")
            10
            >>> config.get("app.debug", False)
            False
        """
        keys = key.split('.')
        current = self._config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def get_page_config(self, page_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific page.

        Args:
            page_name: Name of the page (e.g., "ai_options_agent")

        Returns:
            Dictionary of page configuration

        Examples:
            >>> config = ConfigManager()
            >>> page_cfg = config.get_page_config("ai_options_agent")
            >>> page_cfg["default_dte_range"]
            [20, 45]
        """
        return self.get(f"pages.{page_name}", {})

    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            feature_name: Name of the feature flag

        Returns:
            True if enabled, False otherwise

        Examples:
            >>> config = ConfigManager()
            >>> config.is_feature_enabled("enable_ai_reasoning")
            True
            >>> config.is_feature_enabled("enable_auto_trading")
            False
        """
        return bool(self.get(f"features.{feature_name}", False))

    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific service.

        Args:
            service_name: Name of the service (e.g., "robinhood", "llm")

        Returns:
            Dictionary of service configuration

        Examples:
            >>> config = ConfigManager()
            >>> rh_cfg = config.get_service_config("robinhood")
            >>> rh_cfg["rate_limit"]
            60
        """
        return self.get(f"services.{service_name}", {})

    def get_llm_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific LLM provider.

        Args:
            provider_name: Name of the LLM provider (e.g., "claude", "openai")

        Returns:
            Dictionary of provider configuration

        Examples:
            >>> config = ConfigManager()
            >>> claude_cfg = config.get_llm_provider_config("claude")
            >>> claude_cfg["model"]
            'claude-sonnet-4-5'
        """
        return self.get(f"services.llm.providers.{provider_name}", {})

    def get_cache_ttl(self, cache_type: str = "default") -> int:
        """
        Get cache TTL for a specific cache type.

        Args:
            cache_type: Type of cache (default, short, medium, long, very_long)

        Returns:
            TTL in seconds

        Examples:
            >>> config = ConfigManager()
            >>> config.get_cache_ttl("short")
            60
            >>> config.get_cache_ttl("long")
            3600
        """
        return int(self.get(f"cache.{cache_type}_ttl", 300))

    def reload_config(self) -> None:
        """
        Reload configuration from files.
        Useful for hot-reloading without restarting the application.

        Returns:
            True if reloaded, False if no changes detected
        """
        with self._lock:
            # Check if any config files have been modified
            needs_reload = False

            for config_path, old_mtime in self._file_mtimes.items():
                if config_path.exists():
                    new_mtime = config_path.stat().st_mtime
                    if new_mtime > old_mtime:
                        needs_reload = True
                        break

            if not needs_reload:
                logger.debug("No config changes detected")
                return False

            # Reload all configs
            logger.info("Reloading configuration files...")
            self._config.clear()
            self._load_all_configs()
            logger.info("Configuration reloaded successfully")
            return True

    def get_all_config(self) -> Dict[str, Any]:
        """
        Get the entire configuration dictionary.

        Returns:
            Complete configuration dictionary

        Warning:
            This returns a reference to the internal config.
            Do not modify it directly.
        """
        return self._config

    def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate configuration for required fields and correct types.

        Returns:
            Tuple of (is_valid, list of error messages)

        Examples:
            >>> config = ConfigManager()
            >>> is_valid, errors = config.validate_config()
            >>> if not is_valid:
            ...     for error in errors:
            ...         print(error)
        """
        errors = []

        # Required sections
        required_sections = ["app", "database", "cache", "pages", "features", "services"]
        for section in required_sections:
            if section not in self._config:
                errors.append(f"Missing required section: {section}")

        # Validate app section
        if "app" in self._config:
            app = self._config["app"]
            if "name" not in app or not isinstance(app.get("name"), str):
                errors.append("app.name must be a string")
            if "version" not in app or not isinstance(app.get("version"), str):
                errors.append("app.version must be a string")

        # Validate database section
        if "database" in self._config:
            db = self._config["database"]
            if "pool_max" in db and not isinstance(db["pool_max"], int):
                errors.append("database.pool_max must be an integer")
            if "pool_min" in db and not isinstance(db["pool_min"], int):
                errors.append("database.pool_min must be an integer")

        # Validate cache section
        if "cache" in self._config:
            cache = self._config["cache"]
            ttl_keys = ["default_ttl", "short_ttl", "medium_ttl", "long_ttl"]
            for ttl_key in ttl_keys:
                if ttl_key in cache and not isinstance(cache[ttl_key], int):
                    errors.append(f"cache.{ttl_key} must be an integer")

        return (len(errors) == 0, errors)

    def __repr__(self) -> str:
        return f"<ConfigManager: {len(self._config)} sections loaded>"


# Global instance accessor
_config_instance = None


def get_config() -> ConfigManager:
    """
    Get the global ConfigManager instance.

    Returns:
        Singleton ConfigManager instance

    Examples:
        >>> from src.config_manager import get_config
        >>> config = get_config()
        >>> config.get("app.name")
        'Magnus - Wheel Strategy Dashboard'
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance


# Convenience functions for common operations
def get_page_config(page_name: str) -> Dict[str, Any]:
    """Convenience function to get page configuration"""
    return get_config().get_page_config(page_name)


def is_feature_enabled(feature_name: str) -> bool:
    """Convenience function to check feature flags"""
    return get_config().is_feature_enabled(feature_name)


def get_service_config(service_name: str) -> Dict[str, Any]:
    """Convenience function to get service configuration"""
    return get_config().get_service_config(service_name)


def get_cache_ttl(cache_type: str = "default") -> int:
    """Convenience function to get cache TTL"""
    return get_config().get_cache_ttl(cache_type)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    config = get_config()

    print("=== Configuration Manager Test ===\n")

    # Test basic config access
    print("App Name:", config.get("app.name"))
    print("App Version:", config.get("app.version"))
    print("Database Pool Max:", config.get("database.pool_max"))
    print("Cache Default TTL:", config.get_cache_ttl())
    print()

    # Test page config
    print("AI Options Agent Config:")
    ai_config = config.get_page_config("ai_options_agent")
    print(f"  DTE Range: {ai_config.get('default_dte_range')}")
    print(f"  Delta Range: {ai_config.get('default_delta_range')}")
    print(f"  Max Results: {ai_config.get('max_results')}")
    print()

    # Test feature flags
    print("Feature Flags:")
    print(f"  AI Reasoning: {config.is_feature_enabled('enable_ai_reasoning')}")
    print(f"  Auto Trading: {config.is_feature_enabled('enable_auto_trading')}")
    print(f"  CSV Exports: {config.is_feature_enabled('enable_csv_exports')}")
    print()

    # Test service config
    print("Robinhood Service Config:")
    rh_config = config.get_service_config("robinhood")
    print(f"  Rate Limit: {rh_config.get('rate_limit')}")
    print(f"  Timeout: {rh_config.get('timeout')}")
    print(f"  Retry Attempts: {rh_config.get('retry_attempts')}")
    print()

    # Test LLM provider config
    print("Claude LLM Config:")
    claude_config = config.get_llm_provider_config("claude")
    print(f"  Model: {claude_config.get('model')}")
    print(f"  Max Tokens: {claude_config.get('max_tokens')}")
    print(f"  Temperature: {claude_config.get('temperature')}")
    print()

    # Validate configuration
    print("Configuration Validation:")
    is_valid, errors = config.validate_config()
    if is_valid:
        print("  [OK] Configuration is valid")
    else:
        print("  [ERROR] Configuration has errors:")
        for error in errors:
            print(f"    - {error}")
    print()

    print(config)
