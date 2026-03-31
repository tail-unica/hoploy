"""Hoploy — explainable knowledge-graph recommendation framework."""

from hoploy.core import registry
from hoploy.core.config import Config
from hoploy.core.pipeline import Pipe
from hoploy.core.factory import factory, Request, Response
from hoploy.core.utils import get_logger

logger = get_logger(Config("configs/default.yaml").logging)

__all__ = [
    "registry",
    "logger",
    "Config",
    "Pipe",
    "factory",
    "Request",
    "Response",
]

__version__ = "0.1.0"