from hoploy.components import DefaultHopwiseSequenceScorePostProcessor
from hoploy.registry import SequenceProcessor


@SequenceProcessor("greenfoodlens_sequence_processor")
class GreenFoodLensSequenceProcessor(DefaultHopwiseSequenceScorePostProcessor):
    """GreenFoodLens-specific sequence score post-processor.

    Identical to the default for now — placeholder for future
    domain-specific scoring adjustments.
    """

    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)
