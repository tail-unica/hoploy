from hoploy.sequence_processors.default import DefaultHopwiseSequenceScorePostProcessor
from hoploy.logits_processors.default import DefaultHopwiseLogitsProcessor
from hoploy.core.registry import SequenceProcessor, LogitsProcessor

@SequenceProcessor("autism_sequence_processor")
class AutismSequenceProcessor(DefaultHopwiseSequenceScorePostProcessor):
    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)

@LogitsProcessor("autism_logits_processor")
class AutismLogitsProcessor(DefaultHopwiseLogitsProcessor):
    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)