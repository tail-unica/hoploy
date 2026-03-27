import importlib
import pathlib
import sys
from typing import Type, Dict

from hoploy.model.base import BaseModel
from hoploy.logits_processors.base import BaseLogitsProcessor
from hoploy.sequence_processors.base import BaseSequenceProcessor

class _InternalRegistry:
    """
    Internal registry class to hold all registered components.
    """
    models: Dict[str, Type[BaseModel]] = {}
    logits_processors: Dict[str, Type[BaseLogitsProcessor]] = {}
    sequence_processors: Dict[str, Type[BaseSequenceProcessor]] = {}

    def __repr__(self):
        return (
            f"_InternalRegistry(models={list(self.models.keys())}, "
            f"logits_processors={list(self.logits_processors.keys())}, "
            f"sequence_processors={list(self.sequence_processors.keys())})"
        )

# Singleton instance of the internal registry
_storage = _InternalRegistry()


class PluginRegistry:
    """
    Central registry class to access registered components.
    """
    @staticmethod
    def get(name: str):
        if name in _storage.models:
            return _storage.models[name]
        elif name in _storage.logits_processors:
            return _storage.logits_processors[name]
        elif name in _storage.sequence_processors:
            return _storage.sequence_processors[name]
        else:
            raise ValueError(f"Plugin '{name}' not found in any registry.")


def Model(name: str):
    """ Decorator to register a model class with a given name. """
    def decorator(cls):
        _storage.models[name] = cls
        return cls
    return decorator


def LogitsProcessor(name: str):
    """ Decorator to register a logits processor class with a given name. """
    def decorator(cls):
        _storage.logits_processors[name] = cls
        return cls
    return decorator


def SequenceProcessor(name: str):
    """ Decorator to register a sequence processor class with a given name. """
    def decorator(cls):
        _storage.sequence_processors[name] = cls
        return cls
    return decorator


def load_plugin(plugin_config: dict):
    """
    Scans the directory for .py files and imports them.
    This triggers the @register decorators inside those files.
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