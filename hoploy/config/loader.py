from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, ListConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

FILTERABLE_SECTIONS = ("model", "logits_processors", "sequence_processor")


class BaseConfig:
    """Base wrapper around OmegaConf config objects with runtime updates."""

    def __init__(self, config: DictConfig):
        self._config = config

    @staticmethod
    def _filter_known_keys(current: DictConfig, incoming: dict[str, Any]) -> dict[str, Any]:
        """Keep only keys that exist in the current config section."""
        filtered: dict[str, Any] = {}
        for key, value in incoming.items():
            if key not in current:
                continue
            current_value = current[key]
            if isinstance(current_value, DictConfig) and isinstance(value, dict):
                filtered[key] = BaseConfig._filter_known_keys(current_value, value)
            else:
                filtered[key] = value
        return filtered

    def update(self, **kwargs: Any) -> "BaseConfig":
        """Return a new config object with allowed keys merged in."""
        filtered = self._filter_known_keys(self._config, kwargs)
        if not filtered:
            return self.__class__(OmegaConf.create(OmegaConf.to_container(self._config, resolve=False)))

        patch = OmegaConf.create(filtered)
        updated = OmegaConf.merge(self._config, patch)
        return self.__class__(updated)

    @property
    def raw(self) -> DictConfig:
        return self._config

    def to_dict(self) -> dict[str, Any]:
        return OmegaConf.to_container(self._config, resolve=True)  # type: ignore[return-value]

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._config, name)
        if isinstance(value, DictConfig):
            return BaseConfig(value)
        return value


class AppConfig(BaseConfig):
    """Concrete app config that can be extended with app-specific helpers."""


def _merge_plugin_configs(config: DictConfig) -> DictConfig:
    """Merge plugin configs (override by key) and filter sections by component name."""
    if "plugin" not in config or not isinstance(config.plugin, DictConfig):
        return config

    for plugin_name, plugin_cfg in config.plugin.items():
        if not isinstance(plugin_cfg, DictConfig):
            continue

        # Load and merge plugin's config.yaml — overrides default values by matching keys
        plugin_path = (PROJECT_ROOT / plugin_cfg.path if "path" in plugin_cfg
                       else PROJECT_ROOT / "plugins" / plugin_name)
        plugin_config_file = plugin_path / "config.yaml"
        if plugin_config_file.exists():
            config = OmegaConf.merge(config, OmegaConf.load(str(plugin_config_file)))

        # Filter each section: keep only entries whose `name` matches the allowed list
        for section in FILTERABLE_SECTIONS:
            allowed = plugin_cfg.get(section)
            if not isinstance(allowed, (list, ListConfig)):
                continue
            allowed_names = {str(n) for n in allowed}
            if section in config and isinstance(config[section], DictConfig):
                config[section] = OmegaConf.create({
                    key: entry for key, entry in config[section].items()
                    if isinstance(entry, DictConfig) and str(entry.get("name", "")) in allowed_names
                })

    return config


@lru_cache(maxsize=8)
def _load_raw_config(config_path: str | None = None) -> DictConfig:
    if config_path:
        config = OmegaConf.load(config_path)
    else:
        # Initialize Hydra only once
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            config = compose(config_name="default")
    
    # Apply plugin configuration merging and filtering
    config = _merge_plugin_configs(config)
    return config


def load_config(config_path: str | None = None, updates: dict[str, Any] | None = None) -> AppConfig:
    # Work on a fresh copy so callers cannot mutate the cached base config.
    base = _load_raw_config(config_path)
    config_copy = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
    config = AppConfig(config_copy)
    if updates:
        config = config.update(**updates)
    return config