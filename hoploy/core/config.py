from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, ListConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

FILTERABLE_SECTIONS = ("wrapper", "logits_processors", "sequence_processor")


def _merge_plugin_configs(config: DictConfig) -> DictConfig:
    """Merge plugin configs and filter sections by component name.

    Loads each plugin's ``config.yaml``, merges it into the main config
    (overriding by key), then filters filterable sections to keep only
    entries whose ``name`` appears in the plugin's allow-list.

    :param config: The base application configuration.
    :type config: DictConfig
    :returns: The merged and filtered configuration.
    :rtype: DictConfig
    """
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
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            config = compose(config_name="default")

    config = _merge_plugin_configs(config)
    return config


class Config:
    """Unified, immutable configuration wrapper around OmegaConf.

    Provides attribute-style access, iteration over child sections,
    and a safe :meth:`update` that only merges known keys.

    Usage::

        config = Config("configs/default.yaml")
        config.model          # section accessor
        config.plugin.raw     # underlying DictConfig
        new_cfg = config.update(other_config)
    """

    def __init__(self, config_path: str | None = None, *, _raw: DictConfig | None = None):
        if _raw is not None:
            self._config = _raw
        else:
            base = _load_raw_config(config_path)
            self._config = OmegaConf.create(OmegaConf.to_container(base, resolve=False))

    # -- mutation helpers ---------------------------------------------------

    @staticmethod
    def _filter_known_keys(current: DictConfig, incoming: dict[str, Any]) -> dict[str, Any]:
        """Keep only keys that already exist in *current*.

        Recursively filters nested dicts so that unknown keys from
        *incoming* are silently dropped.

        :param current: The reference config section.
        :type current: DictConfig
        :param incoming: Key-value pairs to filter.
        :type incoming: dict[str, Any]
        :returns: A dict containing only the keys present in *current*.
        :rtype: dict[str, Any]
        """
        filtered: dict[str, Any] = {}
        for key, value in incoming.items():
            if key not in current:
                continue
            current_value = current[key]
            if isinstance(current_value, DictConfig) and isinstance(value, dict):
                filtered[key] = Config._filter_known_keys(current_value, value)
            else:
                filtered[key] = value
        return filtered

    def update(self, other: "Config | None" = None, **kwargs: Any) -> "Config":
        """Return a new :class:`Config` with allowed keys merged in.

        Only keys already present in ``self`` are kept; unknown keys from
        *other* are silently dropped.

        :param other: Another :class:`Config` to merge (e.g. the request
            coming from the router).  Mutually exclusive with *kwargs*.
        :type other: Config | None
        :param kwargs: Plain key-value overrides (used when *other* is ``None``).
        :returns: A new :class:`Config` with the merged values.
        :rtype: Config
        """
        if other is not None:
            incoming = OmegaConf.to_container(other._config, resolve=True)
        else:
            incoming = kwargs

        filtered = self._filter_known_keys(self._config, incoming)
        if not filtered:
            return Config(_raw=OmegaConf.create(OmegaConf.to_container(self._config, resolve=False)))

        patch = OmegaConf.create(filtered)
        updated = OmegaConf.merge(self._config, patch)
        return Config(_raw=updated)

    # -- accessors ----------------------------------------------------------

    @property
    def raw(self) -> DictConfig:
        """Return the underlying :class:`DictConfig`."""
        return self._config

    def to_dict(self) -> dict[str, Any]:
        """Resolve and return the config as a plain Python dict.

        :returns: A recursively-resolved dict.
        :rtype: dict[str, Any]
        """
        return OmegaConf.to_container(self._config, resolve=True)  # type: ignore[return-value]

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying :class:`DictConfig`.

        :class:`DictConfig` children are wrapped in a new :class:`Config`.
        """
        value = getattr(self._config, name)
        if isinstance(value, DictConfig):
            return Config(_raw=value)
        return value

    def __iter__(self):
        """Yield child sections as :class:`Config` objects.

        Raw (non-dict) values are yielded as-is.
        """
        for value in self._config.values():
            if isinstance(value, DictConfig):
                yield Config(_raw=value)
            else:
                yield value
