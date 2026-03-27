import logging

import numpy as np
import torch
from hopwise.model.logits_processor import ConstrainedLogitsProcessorWordLevel
from hopwise.utils import KnowledgeEvaluationType, PathLanguageModelingTokenType

from hoploy.core.registry import LogitsProcessor
from .base import BaseLogitsProcessor


logger = logging.getLogger(__name__)


@LogitsProcessor("default_logits_processor")
class DefaultHopwiseLogitsProcessor(BaseLogitsProcessor, ConstrainedLogitsProcessorWordLevel):
    def __init__(
        self,
        dataset, # dataset interface from the model wrapper, which contains the tokenizer and other dataset-related information
        cfg, # logit processor configuration from the pipeline configuration
        **kwargs,
    ):
        self.dataset = dataset
        self.tokenized_ckg = dataset.get_tokenized_ckg()
        self.tokenized_used_ids = dataset.get_tokenized_used_ids()
        self.tokenizer = dataset.tokenizer
        self.cfg = cfg

        self.remove_user_tokens_from_sequences = bool(
            getattr(self.cfg, "remove_user_tokens_from_sequences", False)
        )
        self.max_sequence_length = int(getattr(self.cfg, "max_sequence_length", 10))
        tokenized_ui_relation = kwargs.pop("tokenized_ui_relation", self.get_relation_id(self.dataset.ui_relation))

        mask_cache_size = int(kwargs.pop("mask_cache_size", 3 * 10**4))
        pos_candidates_cache_size = int(kwargs.pop("pos_candidates_cache_size", 1 * 10**5))
        task = kwargs.pop("task", KnowledgeEvaluationType.REC)

        self.tokenized_uids = set(
            [
                vocab[1]
                for vocab in self.tokenizer.get_vocab().items()
                if vocab[0].startswith(PathLanguageModelingTokenType.USER.token)
            ]
        )
        self.tokenized_ui_relation = {tokenized_ui_relation}
        self.previous_recommendations = None
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
        token_id = self.dataset.field2token_id[self.dataset.relation_field].get(relation_name)
        if token_id is None:
            raise ValueError(f"Relation '{relation_name}' not found in dataset field2token_id mapping.")
        relation_token = PathLanguageModelingTokenType.RELATION.token + str(token_id)
        return self.dataset.tokenizer.convert_tokens_to_ids(relation_token)

    def set_previous_recommendations(self, previous_recommendations):
        """
        Set the previous recommendations to be masked in the next generation step.

        :param previous_recommendations: List or set of token IDs to mask
        """
        if previous_recommendations is None:
            self.previous_recommendations = None
            return

        token_ids = set(previous_recommendations)
        vocab_size = len(self.tokenizer)
        valid_token_ids = {tid for tid in token_ids if 0 <= tid < vocab_size}
        invalid_count = len(token_ids) - len(valid_token_ids)

        if invalid_count:
            logger.warning(
                "set_previous_recommendations: %s invalid token IDs skipped (vocab_size=%s)",
                invalid_count,
                vocab_size,
            )

        self.previous_recommendations = valid_token_ids if valid_token_ids else None
        logger.debug("previous_recommendations: %s tokens masked", len(valid_token_ids))

    def process_scores_rec(self, input_ids, idx):
        """Process each score based on input length and update mask list."""
        key = self.get_current_key(input_ids, idx)
        candidate_tokens = self.get_candidates_rec(*key)

        if self.remove_user_tokens_from_sequences:
            candidate_tokens = candidate_tokens - self.tokenized_uids - self.tokenized_ui_relation

        return key, list(candidate_tokens)

    def get_candidates_rec(self, key1, key2=None):
        """
        Return relation/entity candidates that correspond to real graph edges.

        :param key1: current head token id
        :param key2: relation token id (when requesting entity tails)
        """
        if key1 in self.tokenized_ckg:
            if key2 is not None and key2 in self.tokenized_ckg[key1]:
                return set(self.tokenized_ckg[key1][key2])
            return set(self.tokenized_ckg[key1].keys())
        return set()

    def __call__(self, input_ids, scores):
        """
        Process logits and apply the active recommendation constraints.
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

        return scores

    def config(self, **kwargs):
        """Configure runtime fields while keeping tokenizer-level logic in this base class."""
        self.set_previous_recommendations(kwargs.get("previous_recommendations"))
        return self