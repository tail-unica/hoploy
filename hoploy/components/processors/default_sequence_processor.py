import torch

from hoploy import logger
from hoploy.core.registry import SequenceProcessor
from hoploy.components.processors.base import BaseSequenceProcessor
from hoploy.core.utils import hopwise_encode, hopwise_decode

from hopwise.model.sequence_postprocessor import CumulativeSequenceScorePostProcessor
from hopwise.utils import PathLanguageModelingTokenType


@SequenceProcessor(name="default_cumulative_sequence_processor")
class DefaultHopwiseSequenceScorePostProcessor(BaseSequenceProcessor, CumulativeSequenceScorePostProcessor):
    """Post-processor for zero-shot cumulative sequence scoring.

    Normalises beam-search scores, ranks sequences, and extracts
    ``(user, item, score, decoded_path)`` tuples.

    :param dataset: The Hopwise dataset instance.
    :param cfg: Processor configuration section.
    :type cfg: ~hoploy.core.config.Config
    """

    def __init__(self, dataset, cfg, **kwargs):
        self.dataset = dataset
        self.tokenizer = dataset.tokenizer
        self.item_num = dataset.item_num
        self.topk = int(getattr(cfg, "topk", 10))

    def handle(self, request):
        """No-op — override in plugins to adjust scoring per request.

        :param request: A :class:`~hoploy.core.config.Config` wrapping the
            API payload.
        :type request: Config
        :returns: ``self`` for chaining.
        """
        return self

    def get_sequences(self, generation_outputs, user_num=1, max_new_tokens=24, previous_recommendations=None):
        """Score, sort, and parse generated sequences.

        :param generation_outputs: Raw output dict from ``model.generate``.
        :param user_num: Number of distinct users in the batch.
        :type user_num: int
        :param max_new_tokens: Maximum new tokens generated per sequence.
        :type max_new_tokens: int
        :param previous_recommendations: Token IDs to exclude.
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

        valid_sequences_mask = torch.logical_not(torch.isfinite(normalized_sequences_scores))
        normalized_sequences_scores = torch.where(valid_sequences_mask, -torch.inf, normalized_sequences_scores)

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

    def parse_sequences(self, user_index, sequences, sequences_scores, previous_recommendations=None):
        """Parse sorted sequences into structured recommendation tuples.

        :param user_index: Per-sequence user indices.
        :param sequences: Sorted generated token-ID sequences.
        :param sequences_scores: Corresponding cumulative scores.
        :param previous_recommendations: Token IDs to exclude.
        :type previous_recommendations: list[int] | None
        :returns: A ``(scores_tensor, list_of_tuples)`` tuple where each
            entry is ``[user_idx, item_id, score, decoded_path]``.
        :rtype: tuple
        """
        user_num = user_index.unique().numel()
        scores = torch.full((user_num, self.item_num), -torch.inf)
        user_topk_sequences = list()

        for batch_uidx, sequence, sequence_score in zip(user_index, sequences, sequences_scores):
            parsed_seq = self._parse_single_sequence(
                scores, batch_uidx, sequence, previous_recommendations=previous_recommendations
            )
            if parsed_seq is None:
                continue
            recommended_item, decoded_seq = parsed_seq

            scores[batch_uidx, recommended_item] = sequence_score
            user_topk_sequences.append([batch_uidx, recommended_item, sequence_score.item(), decoded_seq])

        return scores, user_topk_sequences

    def _parse_single_sequence(self, scores, batch_uidx, sequence, previous_recommendations=None):
        """Validate and parse a single generated sequence.

        :param scores: Running scores tensor (modified in-place).
        :param batch_uidx: User index for this sequence.
        :param sequence: Token-ID tensor for one sequence.
        :param previous_recommendations: Token IDs to exclude.
        :type previous_recommendations: list[int] | None
        :returns: ``(recommended_item_id, decoded_token_list)`` or ``None``
            if the sequence is invalid.
        :rtype: tuple | None
        """
        previous_recommendations = previous_recommendations or []

        seq = self.tokenizer.decode(sequence).split(" ")
        seq = list(filter(lambda x: x != self.tokenizer.pad_token, seq))

        # Reject consecutive duplicate tokens
        dup = next(((seq[i], i) for i in range(len(seq) - 1) if seq[i] == seq[i + 1]), None)
        if dup is not None:
            return None

        recommended_token = seq[-1]
        if (
            not recommended_token.startswith(PathLanguageModelingTokenType.ITEM.token)
            or recommended_token == self.tokenizer.pad_token
        ):
            return None

        recommended_item = int(recommended_token[1:])

        if torch.isfinite(scores[batch_uidx, recommended_item]) or recommended_item in previous_recommendations:
            return None

        return recommended_item, seq

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
