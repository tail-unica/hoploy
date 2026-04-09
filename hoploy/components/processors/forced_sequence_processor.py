"""Forced-path sequence post-processor for Hopwise generation pipelines."""

import torch
from hopwise.utils import PathLanguageModelingTokenType

from hoploy import logger
from hoploy.components.processors.default_sequence_processor import (
    DefaultHopwiseSequenceScorePostProcessor,
)
from hoploy.core.registry import SequenceProcessor


@SequenceProcessor("forced_sequence_processor")
class ForcedSequenceScorePostProcessor(DefaultHopwiseSequenceScorePostProcessor):
    """Sequence post-processor that enforces relation-path diversity.

    Extends :class:`DefaultHopwiseSequenceScorePostProcessor` by:

    1. Filtering generated sequences to those that match at least one
       configured forced relation path (``force_paths``).
    2. Boosting the best sequence per path type so that the final
       ranking is diverse across path types.

    Force paths use the same string-relation-name format as
    :class:`~hoploy.components.processors.forced_logits_processor.ForcedLogitsProcessor`
    (e.g. ``["HAS_SENSORY_FEATURE", "HAS_SENSORY_FEATURE_r"]``).

    :param dataset: The Hopwise dataset instance.
    :param cfg: Processor configuration section.
    :type cfg: ~hoploy.core.config.Config
    """

    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)
        self._dataset = dataset

        raw_paths = getattr(cfg, "force_paths", None)
        if raw_paths:
            self._set_force_paths([list(p) for p in raw_paths])
        else:
            self._tokenized_sequence_paths: list[list[str]] = []

    # -- path resolution -------------------------------------------------------

    def _set_force_paths(self, paths: list[list[str]]) -> None:
        """Resolve relation-name paths to tokenized string lists.

        :param paths: List of relation-name sequences, each being a list
            of relation names (e.g. ``["HAS_SENSORY_FEATURE", "HAS_SENSORY_FEATURE_r"]``).
            Use ``"[UI-Relation]"`` for the dataset's user–item relation.
        """
        resolved: list[list[str]] = []
        for path in paths:
            token_path: list[str] = []
            ok = True
            for rel_name in path:
                actual_name = (
                    self._dataset.ui_relation
                    if rel_name == "[UI-Relation]"
                    else rel_name
                )
                rid = self._dataset.field2token_id[self._dataset.relation_field].get(actual_name)
                if rid is None:
                    logger.warning(
                        "Sequence force-path relation '%s' not found in dataset, skipping path",
                        rel_name,
                    )
                    ok = False
                    break
                token_path.append(PathLanguageModelingTokenType.RELATION.token + str(rid))
            if ok:
                resolved.append(token_path)
        self._tokenized_sequence_paths = resolved

    # -- sequence type helpers -------------------------------------------------

    def _get_sequence_type(self, sequence: torch.Tensor) -> int:
        """Return the forced-path index that matches this sequence, or ``-1``.

        Extracts relation tokens (in order) from the decoded sequence and
        checks them against each entry in ``_tokenized_sequence_paths``.

        :param sequence: 1-D token-ID tensor for one generated sequence.
        :returns: Index of the matching path, or ``-1`` if none matches.
        :rtype: int
        """
        relation_prefix = PathLanguageModelingTokenType.RELATION.token
        decoded = self.tokenizer.decode(sequence).split(" ")
        relation_tokens = [t for t in decoded if t.startswith(relation_prefix)]

        for idx, path in enumerate(self._tokenized_sequence_paths):
            if relation_tokens == path:
                return idx
        return -1

    def force_paths_mask(self, sequences: torch.Tensor) -> torch.Tensor:
        """Build a boolean mask of sequences that match a forced path.

        :param sequences: Shape ``(num_sequences, seq_len)`` token-ID tensor.
        :returns: Boolean tensor of shape ``(num_sequences,)`` — ``True``
            when the sequence's relation tokens exactly match a forced path.
        :rtype: torch.BoolTensor
        """
        mask = torch.zeros(sequences.shape[0], dtype=torch.bool, device=sequences.device)
        relation_prefix = PathLanguageModelingTokenType.RELATION.token

        for idx, sequence in enumerate(sequences):
            decoded = self.tokenizer.decode(sequence).split(" ")
            relation_tokens = [t for t in decoded if t.startswith(relation_prefix)]
            for path in self._tokenized_sequence_paths:
                if relation_tokens == path:
                    mask[idx] = True
                    break

        return mask

    def surface_top_sequences_by_type(
        self,
        sequences: torch.Tensor,
        sequences_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Boost the best sequence per path type to ensure diversity.

        Identifies the highest-scoring sequence for each path type and
        artificially raises its score above all others (while preserving
        relative ordering within each type).

        :param sequences: Shape ``(num_sequences, seq_len)`` token-ID tensor.
        :param sequences_scores: Shape ``(num_sequences,)`` score tensor.
        :returns: Modified score tensor with per-type top sequences boosted.
        :rtype: torch.Tensor
        """
        best_by_type: dict[int, tuple[int, float]] = {}  # type_idx -> (seq_idx, score)

        for idx, (sequence, score) in enumerate(zip(sequences, sequences_scores)):
            if not torch.isfinite(score):
                continue
            seq_type = self._get_sequence_type(sequence)
            if seq_type == -1:
                continue
            if seq_type not in best_by_type or score > best_by_type[seq_type][1]:
                best_by_type[seq_type] = (idx, score.item())

        if not best_by_type:
            return sequences_scores

        finite_mask = torch.isfinite(sequences_scores)
        if not finite_mask.any():
            return sequences_scores

        max_score = sequences_scores[finite_mask].max()
        modified_scores = sequences_scores.clone()
        sorted_types = sorted(best_by_type.items(), key=lambda x: x[1][1], reverse=True)

        for rank, (seq_type, (seq_idx, original_score)) in enumerate(sorted_types):
            boosted = max_score + len(sorted_types) - rank
            logger.debug(
                "Boosting sequence idx=%d (type=%d, path=%s): %.4f -> %.4f",
                seq_idx,
                seq_type,
                self._tokenized_sequence_paths[seq_type]
                if seq_type < len(self._tokenized_sequence_paths)
                else "?",
                original_score,
                boosted.item(),
            )
            modified_scores[seq_idx] = boosted

        return modified_scores

    # -- get_sequences override ------------------------------------------------

    def get_sequences(self, generation_outputs, user_num=1, max_new_tokens=24, previous_recommendations=None):
        """Score, sort, and parse sequences with forced-path filtering and diversity.

        When ``force_paths`` are configured:

        1. Sequences not matching any forced path are given ``-inf`` score.
        2. The best sequence per path type is boosted to the top to ensure
           diversity across path patterns.

        :param generation_outputs: Raw output dict from ``model.generate``.
        :param user_num: Number of distinct users in the batch.
        :type user_num: int
        :param max_new_tokens: Maximum new tokens per sequence.
        :type max_new_tokens: int
        :param previous_recommendations: Item IDs to exclude.
        :type previous_recommendations: list[int] | None
        :returns: A ``(scores_tensor, parsed_sequences_list)`` tuple.
        :rtype: tuple
        """
        normalized_scores = self.normalize_tuple(generation_outputs["scores"])
        normalized_sequences_scores = self.calculate_sequence_scores(
            normalized_scores, generation_outputs["sequences"], max_new_tokens=max_new_tokens
        )

        sequences = generation_outputs["sequences"]
        num_return_sequences = sequences.shape[0] // user_num
        batch_user_index = torch.arange(user_num, device=sequences.device).repeat_interleave(num_return_sequences)

        # Mask invalid scores
        invalid_mask = torch.logical_not(torch.isfinite(normalized_sequences_scores))
        normalized_sequences_scores = torch.where(invalid_mask, -torch.inf, normalized_sequences_scores)

        if self._tokenized_sequence_paths:
            normalized_sequences_scores = torch.where(
                self.force_paths_mask(sequences),
                normalized_sequences_scores,
                torch.tensor(-torch.inf, device=normalized_sequences_scores.device),
            )
            normalized_sequences_scores = self.surface_top_sequences_by_type(
                sequences, normalized_sequences_scores
            )

        sorted_indices = normalized_sequences_scores.argsort(descending=True)
        sorted_sequences = sequences[sorted_indices]
        sorted_sequences_scores = normalized_sequences_scores[sorted_indices]
        sorted_batch_user_index = batch_user_index[sorted_indices]

        return self.parse_sequences(
            sorted_batch_user_index,
            sorted_sequences,
            sorted_sequences_scores,
            previous_recommendations=previous_recommendations,
        )

    # -- hook ------------------------------------------------------------------

    def handle(self, request):
        """Configure force paths from the request (if provided).

        :param request: A :class:`~hoploy.core.config.Config` wrapping the
            API payload.
        :returns: ``self`` for chaining.
        """
        force_paths = getattr(request, "force_paths", None)
        if force_paths is not None:
            self._set_force_paths([list(p) for p in force_paths])
        return self
