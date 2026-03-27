from hoploy.model.wrappers.default import DefaultHopwiseWrapper
from hoploy.core.registry import Model

@Model("autism_model")
class AutismWrapper(DefaultHopwiseWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)