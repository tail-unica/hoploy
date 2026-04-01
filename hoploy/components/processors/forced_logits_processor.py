"""Forced-path logits processor for Hopwise generation pipelines."""

import logging

from hopwise.utils import PathLanguageModelingTokenType

from hoploy.components.processors.default_logits_processor import DefaultHopwiseLogitsProcessor
from hoploy.core.registry import LogitsProcessor

logger = logging.getLogger(__name__)


@LogitsProcessor("forced_logits_processor")
class ForcedLogitsProcessor(DefaultHopwiseLogitsProcessor):
    """Graph-constrained processor that forces generation along specific relation paths.

    Extends :class:`DefaultHopwiseLogitsProcessor` by filtering relation
    candidates so only relations belonging to a declared set of forced
    paths are allowed at each generation step.

    Force paths are lists of relation names.  At depth *d* (i.e. after
    *d* relations have already been generated), only relations that
    appear at position *d* in at least one matching force path are
    kept.  Entity selection is not constrained.

    :param dataset: The Hopwise dataset instance.
    :param cfg: Processor configuration section.
    :type cfg: ~hoploy.core.config.Config
    """

    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)

        # Pre-compute set of relation token IDs for fast sequence scanning
        self._relation_token_ids = frozenset(
            v for tok, v in self.tokenizer.get_vocab().items()
            if tok.startswith(PathLanguageModelingTokenType.RELATION.token)
        )

        # Initialise from config (if provided)
        raw_paths = getattr(self.cfg, "force_paths", None)
        if raw_paths:
            self.set_force_paths([list(p) for p in raw_paths])
        else:
            self._force_paths: list[list[int]] = []

    # -- public API ------------------------------------------------------------

    def set_force_paths(self, paths):
        """Set forced relation paths from relation names.

        Each path is a list of relation names as they appear in the
        dataset (e.g. ``["HAS_SENSORY_FEATURE", "HAS_SENSORY_FEATURE_r"]``).
        Use ``"[UI-Relation]"`` to refer to the dataset's user–item
        relation.

        :param paths: Relation-name paths, or ``None`` to clear.
        :type paths: list[list[str]] | None
        """
        if not paths:
            self._force_paths = []
            return

        resolved: list[list[int]] = []
        for path in paths:
            token_ids: list[int] = []
            for rel_name in path:
                actual_name = (
                    self.dataset.ui_relation
                    if rel_name == "[UI-Relation]"
                    else rel_name
                )
                try:
                    token_ids.append(self.get_relation_id(actual_name))
                except (ValueError, KeyError):
                    logger.warning(
                        "Force-path relation '%s' not found, skipping entire path",
                        rel_name,
                    )
                    token_ids = None
                    break
            if token_ids is not None:
                resolved.append(token_ids)
        self._force_paths = resolved

    # -- internals -------------------------------------------------------------

    def _extract_relation_sequence(self, token_ids):
        """Extract relation token IDs from a single beam's input sequence.

        :param token_ids: 1-D tensor of token IDs for one beam.
        :returns: Ordered list of relation token IDs already generated.
        :rtype: list[int]
        """
        return [
            tid.item() for tid in token_ids
            if tid.item() in self._relation_token_ids
        ]

    def _get_allowed_relations(self, candidates, relation_sequence):
        """Return the subset of *candidates* allowed by the forced paths.

        :param candidates: Relation token IDs returned by the graph.
        :type candidates: set[int]
        :param relation_sequence: Relations generated so far (from
            :meth:`_extract_relation_sequence`).
        :type relation_sequence: list[int]
        :returns: Allowed relation token IDs (may be empty).
        :rtype: set[int]
        """
        depth = len(relation_sequence)
        allowed: set[int] = set()
        for path in self._force_paths:
            if depth < len(path):
                if path[:depth] == relation_sequence and path[depth] in candidates:
                    allowed.add(path[depth])
        return allowed

    # -- hook override ---------------------------------------------------------

    def process_scores_rec(self, input_ids, idx):
        """Extend candidate selection with forced-path relation filtering.

        When ``force_paths`` are set and the model is selecting a
        relation (single-element key), only relations that match the
        forced paths at the current depth are kept.
        """
        key = self.get_current_key(input_ids, idx)
        candidate_tokens = self.get_candidates_rec(*key)

        if self.remove_user_tokens_from_sequences:
            candidate_tokens = candidate_tokens - self.tokenized_uids - self.tokenized_ui_relation

        # Force-path filtering applies only to relation selection (len(key) == 1)
        if self._force_paths and len(key) == 1:
            relation_seq = self._extract_relation_sequence(input_ids[idx])
            allowed = self._get_allowed_relations(candidate_tokens, relation_seq)
            if allowed:
                candidate_tokens = allowed

        return key, list(candidate_tokens)

    def handle(self, request):
        """Set previous recommendations and optionally update force paths.

        If the request carries a ``force_paths`` attribute it replaces
        the paths set at init time.

        :param request: A :class:`~hoploy.core.config.Config` wrapping the
            API payload.
        :type request: Config
        :returns: ``self`` for chaining.
        """
        super().handle(request)
        force_paths = getattr(request, "force_paths", None)
        if force_paths is not None:
            self.set_force_paths([list(p) for p in force_paths])
        return self
