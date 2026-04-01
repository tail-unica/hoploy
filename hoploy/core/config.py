from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "hoploy" / "configs" / "default.yaml"

FILTERABLE_SECTIONS = ("wrapper", "logits_processors", "sequence_processor")


def _merge_plugin_configs(config: DictConfig, plugin_paths: tuple[str, ...]) -> DictConfig:
    """Merge plugin configs and filter sections by component name.

    Each plugin's ``config.yaml`` must contain a ``plugin:`` section with
    metadata (name, wrapper allow-list, processors, schema, …).  The
    remaining top-level keys (wrapper definitions, processor definitions,
    etc.) are merged into the main config, and filterable sections are
    trimmed to keep only entries listed in the plugin's allow-list.

    :param config: The base application configuration.
    :type config: DictConfig
    :param plugin_paths: Paths to plugin directories (relative to project root).
    :type plugin_paths: tuple[str, ...]
    :returns: The merged and filtered configuration.
    :rtype: DictConfig
    """
    if plugin_paths:
        paths = plugin_paths
    else:
        raw_list = config.get("plugin")
        if isinstance(raw_list, (list, ListConfig)):
            paths = tuple(str(p) for p in raw_list)
        else:
            paths = ()

    merged_plugins: dict[str, Any] = {}

    for plugin_path_str in paths:
        plugin_path = PROJECT_ROOT / str(plugin_path_str)
        plugin_config_file = plugin_path / "config.yaml"
        if not plugin_config_file.exists():
            continue

        plugin_file_config = OmegaConf.load(str(plugin_config_file))

        # Extract plugin metadata from the plugin's own config.yaml
        plugin_meta = plugin_file_config.get("plugin")
        if not isinstance(plugin_meta, DictConfig):
            continue
        plugin_name = str(plugin_meta.get("name", plugin_path.name))

        # Store metadata with path for pipeline / factory consumption
        meta_dict = OmegaConf.to_container(plugin_meta, resolve=False)
        meta_dict["path"] = str(plugin_path_str)
        merged_plugins[plugin_name] = meta_dict

        # Merge non-plugin sections (wrapper defs, processor defs, …)
        merge_sections = OmegaConf.create({
            k: v for k, v in plugin_file_config.items() if k != "plugin"
        })
        config = OmegaConf.merge(config, merge_sections)

        # Filter each section: keep only entries whose `name` matches the allowed list
        for section in FILTERABLE_SECTIONS:
            allowed = plugin_meta.get(section)
            if not isinstance(allowed, (list, ListConfig)):
                continue
            allowed_names = {str(n) for n in allowed}
            if section in config and isinstance(config[section], DictConfig):
                config[section] = OmegaConf.create({
                    key: entry for key, entry in config[section].items()
                    if isinstance(entry, DictConfig) and str(entry.get("name", "")) in allowed_names
                })

    # Replace the raw plugin list/section with the collected metadata dict
    config_dict = OmegaConf.to_container(config, resolve=False)
    config_dict["plugin"] = merged_plugins
    return OmegaConf.create(config_dict)


@lru_cache(maxsize=8)
def _load_raw_config(plugin_paths: tuple[str, ...] = ()) -> DictConfig:
    config = OmegaConf.load(str(DEFAULT_CONFIG_PATH))
    config = _merge_plugin_configs(config, plugin_paths)
    return config


class Config:
    """Unified, immutable configuration wrapper around OmegaConf.

    Provides attribute-style access, iteration over child sections,
    and a safe :meth:`update` that only merges known keys.

    Usage::

        config = Config("plugins/autism")
        config.wrapper        # section accessor
        config.plugin.raw     # underlying DictConfig
        new_cfg = config.update(other_config)

    The path to ``default.yaml`` is fixed in this module.  Each positional
    argument is a plugin directory (relative to the project root) whose
    ``config.yaml`` will be loaded and merged automatically.
    """

    def __init__(self, *plugin_paths: str, _raw: DictConfig | None = None):
        if _raw is not None:
            self._config = _raw
        else:
            base = _load_raw_config(plugin_paths)
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
