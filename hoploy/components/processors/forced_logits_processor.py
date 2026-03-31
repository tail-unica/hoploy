from hoploy.components.processors.default_logits_processor import DefaultRestrictedHopwiseLogitsProcessor
from hoploy.core.registry import LogitsProcessor


@LogitsProcessor("forced_logits_processor")
class ForcedLogitsProcessor(DefaultRestrictedHopwiseLogitsProcessor):
    """Restricted logits processor registered under a dedicated name."""

    ...
