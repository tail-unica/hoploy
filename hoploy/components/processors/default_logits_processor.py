import numpy as np
import torch
from hopwise.model.logits_processor import ConstrainedLogitsProcessorWordLevel
from hopwise.utils import KnowledgeEvaluationType, PathLanguageModelingTokenType

from hoploy import logger
from hoploy.core.registry import LogitsProcessor
from hoploy.components.processors.base import BaseLogitsProcessor
from hoploy.core.utils import hopwise_encode, hopwise_decode


# ---------------------------------------------------------------------------
# Translation cache helpers
# ---------------------------------------------------------------------------

_PLM_TYPES = (
    PathLanguageModelingTokenType.ITEM,
    PathLanguageModelingTokenType.ENTITY,
    PathLanguageModelingTokenType.RELATION,
    PathLanguageModelingTokenType.USER,
)


def _build_translation_cache(tokenizer):
    """Build bidirectional ``tokenizer_id ↔ hopwise_token`` caches.

    Hopwise tokens use the short format (e.g. ``"I7"``, ``"E17"``,
    ``"R3"``, ``"U8"``).

    :returns: ``(tok2hw, hw2tok)`` dicts.
    :rtype: tuple[dict[int, str], dict[str, int]]
    """
    tok2hw: dict[int, str] = {}
    hw2tok: dict[str, int] = {}
    for token_str, token_id in tokenizer.get_vocab().items():
        for plm_type in _PLM_TYPES:
            if token_str.startswith(plm_type.token):
                tok2hw[token_id] = token_str
                hw2tok[token_str] = token_id
                break
    return tok2hw, hw2tok


@LogitsProcessor("default_logits_processor")
class DefaultHopwiseLogitsProcessor(BaseLogitsProcessor, ConstrainedLogitsProcessorWordLevel):
    """Graph-constrained logits processor for recommendation generation.

    Masks logits so that only tokens reachable via valid knowledge-graph
    edges are kept.  Optionally masks previously recommended items.

    :param dataset: The Hopwise dataset instance.
    :param cfg: Processor configuration section.
    :type cfg: ~hoploy.core.config.Config
    """

    def __init__(self, dataset, cfg, **kwargs):
        self.dataset = dataset
        self.tokenized_ckg = dataset.get_tokenized_ckg()
        self.tokenizer = dataset.tokenizer
        self.cfg = cfg

        # Allow subclasses (e.g. Restricted) to override these via kwargs
        self.tokenized_used_ids = kwargs.pop(
            "tokenized_used_ids", dataset.get_tokenized_used_ids()
        )
        self.remove_user_tokens_from_sequences = bool(
            getattr(self.cfg, "remove_user_tokens_from_sequences", False)
        )
        max_seq_len = kwargs.pop(
            "max_sequence_length",
            getattr(self.cfg, "max_sequence_length", 10),
        )
        self.max_sequence_length = int(max_seq_len) if max_seq_len is not None else None

        ui_relation = getattr(self.dataset, "ui_relation", None)
        tokenized_ui_relation_default = (
            self.get_relation_id(ui_relation) if ui_relation is not None else None
        )
        tokenized_ui_relation = kwargs.pop(
            "tokenized_ui_relation", tokenized_ui_relation_default
        )

        mask_cache_size = int(kwargs.pop("mask_cache_size", 3 * 10**4))
        pos_candidates_cache_size = int(kwargs.pop("pos_candidates_cache_size", 1 * 10**5))
        task = kwargs.pop("task", KnowledgeEvaluationType.REC)

        self.tokenized_uids = {
            vocab[1]
            for vocab in self.tokenizer.get_vocab().items()
            if vocab[0].startswith(PathLanguageModelingTokenType.USER.token)
        }
        self.tokenized_ui_relation = (
            {tokenized_ui_relation} if tokenized_ui_relation is not None else set()
        )
        self.previous_recommendations = None

        # Bidirectional translation cache: tokenizer_id ↔ hopwise_token
        self._tok2hw, self._hw2tok = _build_translation_cache(self.tokenizer)

        super().__init__(
            self.tokenized_ckg,
            self.tokenized_used_ids,
            self.max_sequence_length,
            self.tokenizer,
            mask_cache_size=mask_cache_size,
            pos_candidates_cache_size=pos_candidates_cache_size,
            task=task,
            **kwargs,
        )

    def get_relation_id(self, relation_name):
        """Map a relation name to its tokenizer token ID.

        :param relation_name: The human-readable relation label.
        :type relation_name: str
        :returns: The integer token ID.
        :rtype: int
        :raises ValueError: If the relation is not present in the dataset.
        """
        token_id = self.dataset.field2token_id[self.dataset.relation_field].get(relation_name)
        if token_id is None:
            raise ValueError(f"Relation '{relation_name}' not found in dataset field2token_id mapping.")
        relation_token = PathLanguageModelingTokenType.RELATION.token + str(token_id)
        return self.dataset.tokenizer.convert_tokens_to_ids(relation_token)

    def _hopwise_ids_to_token_ids(self, hopwise_ids):
        """Convert a list of Hopwise token strings to tokenizer IDs.

        Tokens that resolve to the unknown-token ID are silently skipped.

        :param hopwise_ids: Hopwise tokens (e.g. ``"E17"``, ``"I42"``).
        :type hopwise_ids: list[str]
        :returns: List of valid tokenizer integer IDs.
        :rtype: list[int]
        """
        result = []
        for hw_id in hopwise_ids:
            tid = self.tokenizer.convert_tokens_to_ids(hw_id)
            if tid != self.tokenizer.unk_token_id:
                result.append(tid)
        return result

    def set_previous_recommendations(self, hopwise_ids):
        """Set items that should be masked during generation.

        Accepts Hopwise token strings (e.g. ``"I7"``, ``"I42"``).
        The framework translates them to internal tokenizer IDs
        transparently.

        :param hopwise_ids: Hopwise item token strings, or ``None``
            to clear.
        :type hopwise_ids: list[str] | None
        """
        if not hopwise_ids:
            self.previous_recommendations = None
            return

        valid = set(self._hopwise_ids_to_token_ids(hopwise_ids))
        self.previous_recommendations = valid if valid else None
        logger.debug("previous_recommendations: %s tokens masked", len(valid))

    # -- hook: score_adjustment ------------------------------------------------

    def score_adjustment(self, hopwise_current, hopwise_candidates):
        """Hook to apply custom score adjustments during generation.

        Called for each unique beam state after graph-constraint filtering.
        Override in subclasses to implement custom scoring logic using
        Hopwise-level identifiers (never raw tokenizer IDs).

        Identifiers use the short Hopwise token format: ``"I7"``,
        ``"E17"``, ``"R3"``, ``"U8"``.

        :param hopwise_current: Hopwise token of the current graph node.
        :type hopwise_current: str
        :param hopwise_candidates: Hopwise tokens of the graph-valid
            candidate nodes.
        :type hopwise_candidates: list[str]
        :returns: Dict of ``{hopwise_token: score_delta}``.  Use
            ``float('-inf')`` to hard-ban a candidate, negative values to
            penalise, positive to boost.  Candidates not present in the
            returned dict keep their original score.
        :rtype: dict[str, float]
        """
        return {}

    def _apply_score_adjustments(self, unique_input_ids, input_ids_inv, full_mask, scores):
        """Invoke :meth:`score_adjustment` for each unique input and apply deltas."""
        # Skip entirely when the hook is not overridden (zero overhead)
        if type(self).score_adjustment is DefaultHopwiseLogitsProcessor.score_adjustment:
            return

        for idx in range(unique_input_ids.shape[0]):
            allowed_indices = np.where(~full_mask[idx])[0]
            if len(allowed_indices) == 0:
                continue

            current_tok_id = unique_input_ids[idx, -1].item()
            hw_current = self._tok2hw.get(current_tok_id)
            if hw_current is None:
                continue

            hw_candidates = [
                self._tok2hw[int(t)]
                for t in allowed_indices
                if int(t) in self._tok2hw
            ]

            adjustments = self.score_adjustment(hw_current, hw_candidates)
            if not adjustments:
                continue

            beam_rows = np.where(input_ids_inv == idx)[0]
            for hw_id, delta in adjustments.items():
                tok_id = self._hw2tok.get(hw_id)
                if tok_id is None:
                    continue
                if delta == float("-inf"):
                    scores[beam_rows, tok_id] = -torch.inf
                else:
                    scores[beam_rows, tok_id] += delta

    def process_scores_rec(self, input_ids, idx):
        """Return the current key and its candidate tokens for a single input.

        :param input_ids: Batch of input token ID tensors.
        :param idx: Index into the unique-inputs array.
        :type idx: int
        :returns: A ``(key, candidate_tokens)`` pair.
        :rtype: tuple
        """
        key = self.get_current_key(input_ids, idx)
        candidate_tokens = self.get_candidates_rec(*key)

        if self.remove_user_tokens_from_sequences:
            candidate_tokens = candidate_tokens - self.tokenized_uids - self.tokenized_ui_relation

        return key, list(candidate_tokens)

    def get_candidates_rec(self, key1, key2=None):
        """Return relation / entity candidates for real graph edges.

        :param key1: Current head token ID.
        :type key1: int
        :param key2: Relation token ID (when requesting entity tails).
        :type key2: int | None
        :returns: Set of candidate token IDs.
        :rtype: set[int]
        """
        if key1 in self.tokenized_ckg:
            if key2 is not None and key2 in self.tokenized_ckg[key1]:
                return set(self.tokenized_ckg[key1][key2])
            return set(self.tokenized_ckg[key1].keys())
        return set()

    def __call__(self, input_ids, scores):
        """Apply graph constraints, previous-recommendation masking, and score adjustments.

        :param input_ids: Current beam-search input IDs.
        :param scores: Raw logits to modify in-place.
        :returns: The modified scores tensor.
        """
        current_len = input_ids.shape[-1]
        has_bos_token = self.is_bos_token_in_input(input_ids)

        last_n_tokens = 2 if self.is_next_token_entity(input_ids) else 1
        _, input_ids_indices, input_ids_inv = np.unique(
            input_ids.cpu().numpy()[:, -last_n_tokens:],
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        unique_input_ids = input_ids[input_ids_indices]

        full_mask = np.zeros((unique_input_ids.shape[0], len(self.tokenizer)), dtype=bool)
        for idx in range(unique_input_ids.shape[0]):
            if current_len > 2 and (
                self.tokenizer.decode(unique_input_ids[idx, -1]).startswith(PathLanguageModelingTokenType.ITEM.token)
                or unique_input_ids[idx, -1] == self.tokenizer.pad_token_id
            ):
                banned_mask = np.ones(len(self.tokenizer), dtype=bool)
            else:
                try:
                    key, candidate_tokens = self.process_scores_rec(unique_input_ids, idx)
                    banned_mask = self.get_banned_mask(key, candidate_tokens)
                except Exception as exc:
                    logger.warning(
                        "Could not process scores for input idx %s (last token: '%s'): %s. "
                        "Banning all tokens except pad for this input.",
                        idx,
                        self.tokenizer.decode(unique_input_ids[idx, -1]),
                        exc,
                    )
                    banned_mask = np.ones(len(self.tokenizer), dtype=bool)
                    banned_mask[self.tokenizer.pad_token_id] = False

            if self.previous_recommendations:
                try:
                    prev_recs_array = np.array(list(self.previous_recommendations), dtype=np.int64)
                    banned_mask[prev_recs_array] = True
                except (IndexError, ValueError) as exc:
                    logger.error("Failed to apply previous_recommendations mask: %s", exc)

            if banned_mask.all():
                banned_mask[self.tokenizer.pad_token_id] = False

            full_mask[idx] = banned_mask

        if current_len < self.max_sequence_length - 1 - has_bos_token:
            scores[full_mask[input_ids_inv]] = -torch.inf
        else:
            scores[full_mask] = -torch.inf

        self._apply_score_adjustments(unique_input_ids, input_ids_inv, full_mask, scores)

        return scores

    def handle(self, request):
        """Set previous-recommendation mask from the request.

        :param request: A :class:`~hoploy.core.config.Config` wrapping the
            API payload.
        :type request: Config
        :returns: ``self`` for chaining.
        """
        self.set_previous_recommendations(getattr(request, "previous_recommendations", None))
        return self

    def encode(self, value, token_type):
        """Encode a dataset value to a Hopwise token string.

        :param value: The raw dataset value.
        :param token_type: Token prefix string.
        :returns: The encoded token.
        :rtype: str
        """
        return hopwise_encode(self.dataset, value, token_type)

    def decode(self, token, **kwargs):
        """Decode a Hopwise token string back to a dataset value.

        :param token: The encoded token string.
        :param kwargs: Pass ``real_token=True`` to resolve item tokens
            to their human-readable name.
        :returns: The decoded dataset value.
        :rtype: str
        """
        return hopwise_decode(self.dataset, token, real_token=kwargs.get("real_token"))
