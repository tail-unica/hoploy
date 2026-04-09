import importlib
import pathlib
import sys


class _InternalRegistry:
    """Central store for all registered components."""

    def __init__(self):
        self.wrappers: dict[str, type] = {}
        self.logits_processors: dict[str, type] = {}
        self.sequence_processors: dict[str, type] = {}

    def __repr__(self):
        return (
            f"_InternalRegistry(wrappers={list(self.wrappers.keys())}, "
            f"logits_processors={list(self.logits_processors.keys())}, "
            f"sequence_processors={list(self.sequence_processors.keys())})"
        )

# Singleton instance of the internal registry
_storage = _InternalRegistry()


class PluginRegistry:
    """Central registry for looking up registered components by name.

    Components are registered via the :func:`Wrapper`,
    :func:`LogitsProcessor` and :func:`SequenceProcessor` decorators.
    """
    @staticmethod
    def get(name: str):
        """Retrieve a registered component class by *name*.

        :param name: The registration name of the component.
        :type name: str
        :returns: The registered class.
        :raises ValueError: If *name* is not found in any registry.
        """
        if name in _storage.wrappers:
            return _storage.wrappers[name]
        elif name in _storage.logits_processors:
            return _storage.logits_processors[name]
        elif name in _storage.sequence_processors:
            return _storage.sequence_processors[name]
        else:
            raise ValueError(f"Plugin '{name}' not found in any registry.")


def Wrapper(name: str):
    """Decorator to register a model wrapper class under *name*.

    :param name: Unique registration key.
    :type name: str
    """
    def decorator(cls):
        _storage.wrappers[name] = cls
        return cls
    return decorator


def LogitsProcessor(name: str):
    """Decorator to register a logits processor class under *name*.

    :param name: Unique registration key.
    :type name: str
    """
    def decorator(cls):
        _storage.logits_processors[name] = cls
        return cls
    return decorator


def SequenceProcessor(name: str):
    """Decorator to register a sequence processor class under *name*.

    :param name: Unique registration key.
    :type name: str
    """
    def decorator(cls):
        _storage.sequence_processors[name] = cls
        return cls
    return decorator


def load_plugin(plugin_config: dict):
    """Import all Python modules in a plugin directory.

    Importing the modules triggers the ``@Wrapper``, ``@LogitsProcessor``
    and ``@SequenceProcessor`` registration decorators.

    :param plugin_config: A dict with ``"path"`` or ``"name"`` keys
        pointing to the plugin directory.
    :type plugin_config: dict
    """
    if plugin_config.get("path"):
        plugin_path = pathlib.Path(plugin_config.get("path"))
    else:
        plugin_path = pathlib.Path("plugins") / plugin_config.get("name", "")
    
    # Ensure the parent directory is in sys.path so modules are importable
    if str(plugin_path.parent) not in sys.path:
        sys.path.append(str(plugin_path.parent))

    modules_to_import = []

    # Legacy support: import modules placed at plugin root.
    for file in plugin_path.glob("*.py"):
        if file.name == "__init__.py":
            continue
        modules_to_import.append(f"{plugin_path.name}.{file.stem}")

    # New layout: import every module inside typed component folders.
    for component_dir in ("models", "logits_processors", "sequence_processors"):
        target_dir = plugin_path / component_dir
        if not target_dir.is_dir():
            continue

        for file in target_dir.glob("*.py"):
            if file.name == "__init__.py":
                continue
            modules_to_import.append(f"{plugin_path.name}.{component_dir}.{file.stem}")

    # Dynamic imports trigger decorator-based registration.
    for module_name in modules_to_import:
        importlib.import_module(module_name)