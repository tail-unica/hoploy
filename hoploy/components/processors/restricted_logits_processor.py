"""Restriction-based logits processor for Hopwise generation pipelines."""

import logging

import numpy as np
from hopwise.utils import PathLanguageModelingTokenType

from hoploy.core.registry import LogitsProcessor
from hoploy.components.processors.default_logits_processor import DefaultHopwiseLogitsProcessor

logger = logging.getLogger(__name__)


@LogitsProcessor("restricted_logits_processor")
class RestrictedHopwiseLogitsProcessor(DefaultHopwiseLogitsProcessor):
    """Logits processor that applies hard and soft entity / item restrictions.

    Operates as a separate processor in a HuggingFace ``LogitsProcessorList``
    alongside :class:`~hoploy.components.processors.DefaultHopwiseLogitsProcessor`.
    This processor handles only the restriction masking; graph-constrained
    generation is delegated to the default one.

    :param dataset: The Hopwise dataset instance.
    :param cfg: Processor configuration section.
    :type cfg: ~hoploy.core.config.Config
    """

    def __init__(self, dataset, cfg, **kwargs):
        propagate_connected_entities = bool(getattr(cfg, "propagate_connected_entities", True))

        # This processor does not perform graph-constrained generation itself;
        # pass None so ConstrainedLogitsProcessorWordLevel skips those internals.
        kwargs.setdefault("tokenized_used_ids", None)
        kwargs.setdefault("max_sequence_length", None)

        super().__init__(dataset, cfg, **kwargs)

        self.propagate_connected_entities = propagate_connected_entities
        self.tokenized_entities = [
            v for tok, v in self.tokenizer.get_vocab().items()
            if tok.startswith(PathLanguageModelingTokenType.ENTITY.token)
            or tok.startswith(PathLanguageModelingTokenType.ITEM.token)
        ]

        self.current_restricted_candidates = []
        self.current_hard_restrictions = []
        self.current_soft_restrictions = []

    def set_restrictions(self, restricted_candidates=None, hard_restrictions=None, soft_restrictions=None):
        """Set entity / item restriction lists for generation.

        :param restricted_candidates: Token IDs to restrict as candidates.
        :type restricted_candidates: list[int] | None
        :param hard_restrictions: Token IDs to ban unconditionally.
        :type hard_restrictions: list[int] | None
        :param soft_restrictions: Token IDs to ban unless doing so would
            block all tokens.
        :type soft_restrictions: list[int] | None
        """
        self.current_restricted_candidates = restricted_candidates or []
        self.current_hard_restrictions = hard_restrictions or []
        self.current_soft_restrictions = soft_restrictions or []

    def clear_restrictions(self):
        """Remove all active restrictions."""
        self.current_restricted_candidates = []
        self.current_hard_restrictions = []
        self.current_soft_restrictions = []

    def _extract_connected_entities(self, token_id):
        """Return entity token IDs reachable from *token_id* via any relation.

        :param token_id: The source token ID.
        :type token_id: int
        :returns: Set of connected entity token IDs.
        :rtype: set[int]
        """
        connected = set()
        for entity_set in self.tokenized_ckg.get(token_id, {}).values():
            connected.update(entity_set if isinstance(entity_set, (list, set)) else [entity_set])
        return connected

    def _gen_keepmask_restricted_candidates(self):
        """Generate a boolean mask banning everything except shared candidates.

        :returns: A NumPy boolean array of length ``vocab_size``.
        :rtype: numpy.ndarray
        """
        mask = np.zeros(len(self.tokenizer), dtype=bool)
        if not self.current_restricted_candidates:
            return mask

        mask[self.tokenized_entities] = True
        shared = self._extract_connected_entities(self.current_restricted_candidates[0])
        for r_candidate in self.current_restricted_candidates[1:]:
            if r_candidate in self.tokenized_ckg:
                shared &= self._extract_connected_entities(r_candidate)
        mask[self.current_restricted_candidates] = False
        mask[list(shared)] = False
        return mask

    def _gen_banmask_from_key(self, token_id):
        """Generate a ban mask for *token_id* and optionally its neighbours.

        :param token_id: The token ID to ban.
        :type token_id: int
        :returns: A NumPy boolean array of length ``vocab_size``.
        :rtype: numpy.ndarray
        """
        mask = np.zeros(len(self.tokenizer), dtype=bool)
        mask[token_id] = True
        if token_id in self.tokenized_ckg and self.propagate_connected_entities:
            mask[list(self._extract_connected_entities(token_id))] = True
        return mask

    def __call__(self, input_ids, scores):
        """Apply restriction masks to generation scores.

        :param input_ids: Current beam-search input IDs.
        :param scores: Raw logits to modify in-place.
        :returns: The modified scores tensor.
        """
        full_mask = self._gen_keepmask_restricted_candidates()

        if np.all(full_mask):
            logger.warning("Restriction mask blocks all tokens, skipping restrictions.")
            return scores

        for h_rest in self.current_hard_restrictions:
            full_mask = np.logical_or(full_mask, self._gen_banmask_from_key(h_rest))

        if np.all(full_mask):
            logger.warning("Hard restrictions block all tokens, skipping restrictions.")
            return scores

        sorted_soft = sorted(
            self.current_soft_restrictions,
            key=lambda k: len(self.tokenized_ckg.get(k, [])),
            reverse=True,
        )
        for s_rest in sorted_soft:
            soft_mask = np.logical_or(full_mask, self._gen_banmask_from_key(s_rest))
            if np.all(soft_mask):
                break
            full_mask = soft_mask

        scores[:, full_mask] = -np.inf
        return scores

    def handle(self, request):
        """Clear restrictions (override in plugins for custom logic).

        :param request: A :class:`~hoploy.core.config.Config` wrapping the
            API payload.
        :type request: Config
        :returns: ``self`` for chaining.
        """
        self.clear_restrictions()
        return self
