from hoploy.core import registry
from hoploy.config import loader
from hoploy.core.utils import get_logger

logger = get_logger(loader.load_config("configs/default.yaml").logging)

__all__ = ["registry", "logger"]

__version__ = "0.1.0"