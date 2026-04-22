import os

from hoploy.core.config import Config
from hoploy.core.pipeline import Pipe
from hoploy.core.factory import factory

_raw = os.environ.get("HOPLOY_PLUGINS", "plugins/hummus")
_plugins = [p.strip() for p in _raw.split(",") if p.strip()]

config = Config(*_plugins)
pipeline = Pipe(config)
app = factory(pipeline, config)
