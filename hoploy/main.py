from hoploy.core.config import Config
from hoploy.core.pipeline import Pipe
from hoploy.core.factory import factory

config = Config("configs/default.yaml")
pipeline = Pipe(config)
app = factory(pipeline, config)
