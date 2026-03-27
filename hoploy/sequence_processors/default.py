from hoploy.core.registry import SequenceProcessor
from hoploy.sequence_processors.base import BaseSequenceProcessor
from hoploy import logger

from hopwise.model.sequence_postprocessor import CumulativeSequenceScorePostProcessor
from hopwise.utils import PathLanguageModelingTokenType
import torch

@SequenceProcessor(name="default_cumulative_sequence_processor")
class DefaultHopwiseSequenceScorePostProcessor(BaseSequenceProcessor, CumulativeSequenceScorePostProcessor):
    """
    Post-processor for zero-shot cumulative sequence scoring.
    This processor applies a cumulative score to sequences based on their relevance and diversity.
    """

    def __init__(
        self,
        dataset,
        cfg,
        **kwargs,
    ):
        self.tokenizer = dataset.tokenizer
        self.item_num = dataset.item_num
        self.topk = int(getattr(cfg, "topk", 10))

    def config(self, **kwargs):
        return self

    def get_sequences(self, generation_outputs, user_num=1, max_new_tokens=24, previous_recommendations=None):
        normalized_scores = self.normalize_tuple(generation_outputs["scores"])
        normalized_sequences_scores = self.calculate_sequence_scores(
            normalized_scores, generation_outputs["sequences"], max_new_tokens=max_new_tokens
        )

        sequences = generation_outputs["sequences"]
        num_return_sequences = sequences.shape[0] // user_num
        batch_user_index = torch.arange(user_num, device=sequences.device).repeat_interleave(num_return_sequences)
        logger.debug(f"get_sequences: {sequences.shape[0]} sequences, user_num={user_num}, max_new_tokens={max_new_tokens}")

        valid_sequences_mask = torch.logical_not(torch.isfinite(normalized_sequences_scores))  # false if finite
        n_nonfinite = valid_sequences_mask.sum().item()
        if n_nonfinite:
            logger.debug(f"get_sequences: {n_nonfinite}/{sequences.shape[0]} sequences have non-finite scores (will be set to -inf)")
        normalized_sequences_scores = torch.where(valid_sequences_mask, -torch.inf, normalized_sequences_scores)

        # TODO: write an interface for custom sequence sorting strategy

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
    
    def parse_sequences(
        self, 
        user_index, 
        sequences, 
        sequences_scores, 
        previous_recommendations=None
     ) -> tuple[torch.Tensor, list]:
        """
        Parses the sequences and their scores into a structured format.
        """
        user_num = user_index.unique().numel()
        scores = torch.full((user_num, self.item_num), -torch.inf)
        user_topk_sequences = list()
        total = len(sequences)

        for batch_uidx, sequence, sequence_score in zip(user_index, sequences, sequences_scores):
            parsed_seq = self._parse_single_sequence(
                scores, batch_uidx, sequence, previous_recommendations=previous_recommendations
            )
            if parsed_seq is None:
                continue
            recommended_item, decoded_seq = parsed_seq

            scores[batch_uidx, recommended_item] = sequence_score
            user_topk_sequences.append([batch_uidx, recommended_item, sequence_score.item(), decoded_seq])

        logger.debug(f"parse_sequences: {len(user_topk_sequences)}/{total} sequences accepted")
        return scores, user_topk_sequences
    
    def _parse_single_sequence(
        self, 
        scores, 
        batch_uidx, 
        sequence, 
        previous_recommendations=None
    ) -> tuple[int, list] | None:
        """
        Parses a single sequence to extract user ID, recommended item, and the decoded sequence.
        """
        previous_recommendations = previous_recommendations or []

        seq = self.tokenizer.decode(sequence).split(" ")
        seq = list(filter(lambda x: x != self.tokenizer.pad_token, seq))
        logger.debug(f"_parse_single_sequence: decoded={seq}")

        # Bug behavior: check for consecutive duplicate tokens
        dup = next(((seq[i], i) for i in range(len(seq) - 1) if seq[i] == seq[i + 1]), None)
        if dup is not None:
            logger.debug(f"_parse_single_sequence: rejected — consecutive duplicate '{dup[0]}' at pos {dup[1]}")
            return None

        recommended_token = seq[-1]
        if (
            not recommended_token.startswith(PathLanguageModelingTokenType.ITEM.token)
            or recommended_token == self.tokenizer.pad_token
        ):
            logger.debug(f"_parse_single_sequence: rejected — last token '{recommended_token}' is not an item token")
            return None

        recommended_item = int(recommended_token[1:])

        if torch.isfinite(scores[batch_uidx, recommended_item]) or recommended_item in previous_recommendations:
            logger.debug(f"_parse_single_sequence: rejected — item {recommended_item} already scored or in previous_recommendations")
            return None

        return recommended_item, seq
