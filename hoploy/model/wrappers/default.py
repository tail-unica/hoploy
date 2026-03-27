import torch
import pathlib
from typing import Any

from hopwise.data import Interaction
from hopwise.utils import PathLanguageModelingTokenType

from hoploy.model.base import BaseModel
from hoploy.core.registry import Model

from hoploy import logger

@Model("default_model")
class DefaultHopwiseWrapper(BaseModel):
    def __init__(self, cfg):
        self.cfg = cfg
        self._generation_kwargs: dict[str, Any] = {}
        self._recommendation_count: int = 5
        self.logits_processors = []
        self.sequence_processor = None

        checkpoint_file = pathlib.Path("checkpoint") / self.cfg.hopwise_checkpoint_file
        logger.info(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(
            self.cfg.hopwise_checkpoint_file,
            map_location=self.cfg.device,
            weights_only=False,
        )
        config = checkpoint["config"]
        config["checkpoint_dir"] = pathlib.Path(self.cfg.hopwise_checkpoint_file).parent
        config["data_path"] = pathlib.Path(self.cfg.dataset)
        config["load_col"]["item"] = ["poi_id", "name"]
        config["train_stage"] = "pretrain"
        config._set_env_behavior()
        
        from hopwise.data.utils import data_preparation
        from hopwise.utils import get_model
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from hopwise.utils import init_seed
        from hopwise.data.utils import create_dataset
    
        logger.info("Initializing model and dataset")

        init_seed(config["seed"], config["reproducibility"])
        self._dataset = create_dataset(config)

        train_data, _, _ = data_preparation(config, self._dataset)
        self.model = get_model(config["model"])(config, train_data.dataset)
        self.model = self.model.to(device=self.cfg.device, dtype=config["weight_precision"])

        hf_checkpoint_file = self.cfg.hopwise_checkpoint_file.replace("hopwise", "huggingface")
        weights = load_file(pathlib.Path(hf_checkpoint_file) / "model.safetensors")
        self.model.load_state_dict(weights, strict=False)

        self._dataset._tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint_file)
        self.model = torch.compile(self.model, mode=self.cfg.compile_mode)

    @property
    def dataset(self):
        return self._dataset

    def _build_generation_kwargs(self, recommendation_count: int, diversity_factor: float) -> dict[str, Any]:
        adjusted_recommendation_count = int(recommendation_count * 4)
        num_beams = int(adjusted_recommendation_count * 1.5) // 2 * 2 + 2
        token_sequence_length = getattr(self.model, "token_sequence_length", None)

        return {
            "max_length": token_sequence_length,
            "min_length": token_sequence_length,
            "paths_per_user": adjusted_recommendation_count,
            "num_beams": num_beams,
            "num_beam_groups": max(2, num_beams // 2),
            "diversity_penalty": diversity_factor,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

    def update_processors(self, logits_processors=None, sequence_processor=None):
        if logits_processors is not None:
            self.logits_processors = [processor for processor in logits_processors if processor is not None]
        if sequence_processor is not None:
            self.sequence_processor = sequence_processor

        # Unwrap torch.compile OptimizedModule to set attributes on the real model
        model = getattr(self.model, '_orig_mod', self.model)
        model.logits_processor_list = self.logits_processors
        if self.sequence_processor is not None:
            model.sequence_postprocessor = self.sequence_processor

        return self
    
    def config(self, **kwargs):
        """Configure runtime generation defaults from request/model parameters."""
        recommendation_count = int(kwargs.get("recommendation_count", 5))
        diversity_factor = float(kwargs.get("diversity_factor", 0.5))
        self._recommendation_count = recommendation_count
        self._generation_kwargs = self._build_generation_kwargs(recommendation_count, diversity_factor)
        return self

    def recommend(self, inputs):
        """
        :param inputs: Hopwise-tokenized input tensors e.g. [I23, R34, I45, ...]

        :return scores: List of recommendation scores corresponding to the recommendations
        :return recommendations: List of recommended item IDs corresponding to the input format (e.g. [I678, I789, ...])
        :return explanations: List of path explanations for each recommendation
        """
        logger.debug(f"recommend: received {len(inputs)} raw inputs: {inputs}")

        inputs = self.dataset.tokenizer(inputs, return_tensors="pt", add_special_tokens=False).to(self.cfg.device)
        inputs = Interaction(inputs.data)
        logger.debug(f"recommend: tokenized input_ids shape={inputs['input_ids'].shape}, decoded={[self.dataset.tokenizer.decode(row) for row in inputs['input_ids']]}")

        valid_inputs_mask = torch.isin(
            inputs["input_ids"][:, 1:], torch.tensor(self.dataset.tokenizer.all_special_ids, device=inputs["input_ids"].device)
        ).squeeze(dim=1)
        if valid_inputs_mask.all():
            logger.error("All input tokens are special tokens. Returning None.")
            return None

        inputs = inputs[torch.logical_not(valid_inputs_mask)]
        logger.debug(f"recommend: {inputs['input_ids'].shape[0]} inputs after special-token filter")

        logger.info(f"Executing generation with {inputs['input_ids'].shape[0]} input samples")
        try:
            outputs = self.model.generate(inputs, **self._generation_kwargs)
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return None

        logger.debug(f"recommend: generated {outputs['sequences'].shape[0]} raw sequences")
        max_new_tokens = self.model.token_sequence_length - inputs["input_ids"].size(1)
        logger.debug(f"recommend: token_sequence_length={self.model.token_sequence_length}, input_len={inputs['input_ids'].size(1)}, max_new_tokens={max_new_tokens}")
        _, sequences = self.model.sequence_postprocessor.get_sequences(
            outputs, max_new_tokens=max_new_tokens
        )

        if not sequences:
            logger.warning("No valid sequences produced by the model.")
            return None

        top_rec_index = sorted(range(len(sequences)), key=lambda i: sequences[i][2], reverse=True)[:self._recommendation_count]
        sequences = [sequences[i] for i in top_rec_index]

        recommendations = [seq[1] for seq in sequences]
        scores = [seq[2] for seq in sequences]
        explanations = [seq[3] for seq in sequences]

        return scores, recommendations, explanations

    def distill(self, **kwargs):
        """Normalize incoming payload and keep request-level metadata."""
        inputs = kwargs.get("input", [])
        logger.debug(f"distill: payload keys={list(kwargs.keys())}, 'input'={inputs}")
        separator = self.dataset.path_token_separator
        raw_inputs = [
            separator.join([
                self.dataset.tokenizer.bos_token,
                self.encode(item, PathLanguageModelingTokenType.ITEM.token)
            ])
            for item in inputs
        ]
        logger.debug(f"Encoded raw inputs: {raw_inputs}")
        return raw_inputs

    def encode(self, value, token_type):
        # Dataset IDs to hopwise IDs
        def item():
            # example: "55" -> "I1"
            token_iid_list = self.dataset.field2id_token[self.dataset.iid_field]
            return {tok: idx for idx, tok in enumerate(token_iid_list)}

        def entity():
            # example: "SensoryFeature.NOISE.2.3" -> "R789"
            token_eid_list = self.dataset.field2id_token[self.dataset.entity_field]
            return {tok: idx for idx, tok in enumerate(token_eid_list)}
        
        def relation():
            # example: "HAS_SENSORY_FEATURE" -> "R1"
            token_rid_list = self.dataset.field2id_token[self.dataset.relation_field]
            return {tok: idx for idx, tok in enumerate(token_rid_list)}

        def user():
            # example: "474" -> "U42"
            token_uid_list = self.dataset.field2id_token[self.dataset.uid_field]
            return {tok: idx for idx, tok in enumerate(token_uid_list)}

        map = {
            PathLanguageModelingTokenType.ITEM.token: item(),
            PathLanguageModelingTokenType.ENTITY.token: entity(),
            PathLanguageModelingTokenType.RELATION.token: relation(),
            PathLanguageModelingTokenType.USER.token: user(),
        }

        return token_type + str(map[token_type][value])
    
    def decode(self, token):
        if token.startswith(PathLanguageModelingTokenType.ITEM.token):
            token = self.dataset.field2id_token[self.dataset.iid_field][int(token[1:])]
        elif token.startswith(PathLanguageModelingTokenType.ENTITY.token):
            token = self.dataset.field2id_token[self.dataset.entity_field][int(token[1:])]
        elif token.startswith(PathLanguageModelingTokenType.RELATION.token):
            token = self.dataset.field2id_token[self.dataset.relation_field][int(token[1:])]
        elif token.startswith(PathLanguageModelingTokenType.USER.token):
            token = self.dataset.field2id_token[self.dataset.uid_field][int(token[1:])]

        return token

    def expand(self, values):
        """Plugin extension point to enrich model outputs."""
        scores, recommendations, explanations = values
        return {
            "scores": scores,
            "recommendations": [
                self.decode(PathLanguageModelingTokenType.ITEM.token + str(r))
                for r in recommendations
            ],
            "explanations": [
                [self.decode(token) for token in explanation]
                for explanation in explanations
            ],
        }