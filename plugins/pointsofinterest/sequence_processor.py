from hoploy.components import ForcedSequenceScorePostProcessor
from hoploy.registry import SequenceProcessor


@SequenceProcessor("poi_sequence_processor")
class PointsOfInterestSequenceProcessor(ForcedSequenceScorePostProcessor):
    """Points of Interest-specific sequence score post-processor.

    Extends :class:`~hoploy.components.processors.forced_sequence_processor.ForcedSequenceScorePostProcessor`
    to apply forced relation-path filtering and per-type diversity boosting
    as configured in the plugin's ``sequence_processor`` config section.
    """

    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)
