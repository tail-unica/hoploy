import pathlib
from typing import Any, final

import torch
from hopwise.data import Interaction
from hopwise.utils import PathLanguageModelingTokenType

from hoploy import logger
from hoploy.components.wrappers.base import BaseWrapper
from hoploy.core.registry import Wrapper
from hoploy.core.utils import hopwise_encode, hopwise_decode

_CHECKPOINTS_DIR = pathlib.Path("/app/checkpoints")
_DATASET_DIR = "/app/dataset"


def _find_hopwise_checkpoint() -> pathlib.Path:
    """Find the first ``hopwise*.pth`` file in ``/app/checkpoints``."""
    matches = sorted(_CHECKPOINTS_DIR.glob("hopwise*.pth"))
    if not matches:
        raise FileNotFoundError(
            f"No hopwise*.pth checkpoint found in {_CHECKPOINTS_DIR}"
        )
    return matches[0]


@Wrapper("default_model")
class DefaultHopwiseWrapper(BaseWrapper):
    """Default wrapper around Hopwise language models.

    Loads a checkpoint, prepares the dataset and model, and exposes
    :meth:`distill`, :meth:`handle`, :meth:`recommend` and :meth:`expand`
    for the pipeline.

    :param cfg: Wrapper configuration section.
    :type cfg: ~hoploy.core.config.Config
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._runtime_cfg = cfg

        # Ensure all default CUDA allocations (including hopwise internals) land
        # on the same device as specified in config, not necessarily cuda:0.
        _device = torch.device(self.cfg.device)
        if _device.type == "cuda":
            torch.cuda.set_device(_device)

        checkpoint_file = _find_hopwise_checkpoint()
        logger.info(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(
            checkpoint_file,
            map_location=self.cfg.device,
            weights_only=False,
        )

        config = checkpoint["config"]

        config["checkpoint_dir"] = str(checkpoint_file.parent)
        config["load_col"]["item"] = list(getattr(self.cfg, "load_col_item", config["load_col"]["item"]))
        config["data_path"] = _DATASET_DIR
        config["device"] = self.cfg.device

        if train_stage := getattr(self.cfg, "train_stage", None):
            config["train_stage"] = train_stage

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
        logger.debug(f"Dataset created with tokenizer vocab size: {len(self._dataset.tokenizer)}")

        train_data, _, _ = data_preparation(config, self._dataset)
        logger.debug(f"Data preparation completed. Train dataset size: {len(train_data.dataset)}")

        self.model = get_model(config["model"])(config, train_data.dataset)
        logger.debug(f"Model instance created: {self.model.__class__.__name__}")

        self.model = self.model.to(device=self.cfg.device, dtype=config["weight_precision"])
        logger.debug(f"Model initialized on device {self.cfg.device} with dtype {config['weight_precision']}")

        hf_checkpoint_file = str(checkpoint_file).replace("hopwise", "huggingface")
        weights = load_file(str(pathlib.Path(hf_checkpoint_file) / "model.safetensors"))
        self.model.load_state_dict(weights, strict=False)

        self._dataset._tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint_file)
        self.model = torch.compile(self.model, mode=self.cfg.compile_mode)
        logger.debug(f"Model compiled with mode {self.cfg.compile_mode}")

    @property
    def dataset(self):
        """The Hopwise dataset instance backing this wrapper."""
        return self._dataset

    # -- pipeline hooks --------------------------------------------------------

    def handle(self, request):
        """Merge request parameters into the wrapper config.

        :param request: A :class:`~hoploy.core.config.Config` wrapping the
            API payload.
        :type request: Config
        :returns: ``self`` for chaining.
        """
        self._runtime_cfg = self.cfg.update(request)
        return self

    def distill(self, request):
        """Convert raw API input items into BOS-prefixed token strings.

        :param request: A :class:`~hoploy.core.config.Config` with an
            ``input`` key listing raw item identifiers.
        :type request: Config
        :returns: List of tokenised input strings.
        :rtype: list[str]
        """
        inputs = getattr(request, "input", [])
        logger.debug(f"distill: 'input'={inputs}")
        separator = self.dataset.path_token_separator
        raw_inputs = [
            separator.join([
                self.dataset.tokenizer.bos_token,
                self.encode(item, PathLanguageModelingTokenType.ITEM.token),
            ])
            for item in inputs
        ]
        logger.debug(f"Encoded raw inputs: {raw_inputs}")
        return raw_inputs

    def expand(self, values, request):
        """Decode model output into a JSON-serialisable response dict.

        :param values: A ``(scores, recommendations, explanations)`` tuple.
        :param request: The original request :class:`~hoploy.core.config.Config`.
        :type request: Config
        :returns: A dict with ``scores``, ``recommendations`` and
            ``explanations`` keys.
        :rtype: dict
        """
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

    # -- generation ------------------------------------------------------------

    def _build_generation_kwargs(self):
        """Derive beam-search parameters from :attr:`_runtime_cfg`.

        :returns: Keyword arguments for ``model.generate()``.
        :rtype: dict[str, Any]
        """
        recommendation_count = int(self._runtime_cfg.recommendation_count)
        diversity_factor = float(self._runtime_cfg.diversity_factor)

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
            # "trust_remote_code": True,
        }

    def update_processors(self, logits_processors=None, sequence_processor=None):
        """Push processor instances directly onto the underlying model.

        :param logits_processors: List of logits processor instances
            (``None`` entries are filtered out).
        :type logits_processors: list | None
        :param sequence_processor: A sequence post-processor instance.
        :returns: ``self`` for chaining.
        """
        model = getattr(self.model, "_orig_mod", self.model)
        if logits_processors is not None:
            model.logits_processor_list = [p for p in logits_processors if p is not None]
        if sequence_processor is not None:
            model.sequence_postprocessor = sequence_processor
        return self

    @final
    def recommend(self, inputs: list[str]) -> tuple | None:
        """Run beam-search generation and return top recommendations.

        .. warning::
            This method is **sealed** — subclasses must not override it.

        :param inputs: Hopwise-tokenized input strings,
            e.g. ``["<s> I23", ...]``.
        :returns: A ``(scores, item_ids, explanations)`` tuple, or
            ``None`` when no valid sequences are produced.
        :rtype: tuple | None
        """
        logger.debug(f"recommend: received {len(inputs)} raw inputs: {inputs}")

        inputs = self.dataset.tokenizer(inputs, return_tensors="pt", add_special_tokens=False).to(self.cfg.device)
        inputs = Interaction(inputs.data)
        logger.debug(
            f"recommend: tokenized input_ids shape={inputs['input_ids'].shape}, "
            f"decoded={[self.dataset.tokenizer.decode(row) for row in inputs['input_ids']]}"
        )

        valid_inputs_mask = torch.isin(
            inputs["input_ids"][:, 1:],
            torch.tensor(self.dataset.tokenizer.all_special_ids, device=inputs["input_ids"].device),
        ).squeeze(dim=1)
        if valid_inputs_mask.all():
            logger.error("All input tokens are special tokens. Returning None.")
            return None

        inputs = inputs[torch.logical_not(valid_inputs_mask)]
        logger.debug(f"recommend: {inputs['input_ids'].shape[0]} inputs after special-token filter")

        generation_kwargs = self._build_generation_kwargs()
        logger.info(f"Executing generation with {inputs['input_ids'].shape[0]} input samples")
        try:
            outputs = self.model.generate(inputs, **generation_kwargs)
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return None

        logger.debug(f"recommend: generated {outputs['sequences'].shape[0]} raw sequences")
        max_new_tokens = self.model.token_sequence_length - inputs["input_ids"].size(1)
        logger.debug(
            f"recommend: token_sequence_length={self.model.token_sequence_length}, "
            f"input_len={inputs['input_ids'].size(1)}, max_new_tokens={max_new_tokens}"
        )
        _, sequences = self.model.sequence_postprocessor.get_sequences(
            outputs, max_new_tokens=max_new_tokens
        )

        if not sequences:
            logger.warning("No valid sequences produced by the model.")
            return None

        recommendation_count = int(self._runtime_cfg.recommendation_count)
        top_rec_index = sorted(
            range(len(sequences)), key=lambda i: sequences[i][2], reverse=True
        )[:recommendation_count]
        sequences = [sequences[i] for i in top_rec_index]

        recommendations = [seq[1] for seq in sequences]
        scores = [seq[2] for seq in sequences]
        explanations = [seq[3] for seq in sequences]

        return scores, recommendations, explanations

    # -- encoding helpers ------------------------------------------------------

    @final
    def encode(self, value: Any, token_type: str) -> str:
        """Encode a dataset value to a Hopwise token string.

        .. warning::
            This method is **sealed** — subclasses must not override it.

        :param value: The raw dataset value.
        :param token_type: Token prefix string.
        :returns: The encoded token.
        :rtype: str
        """
        return hopwise_encode(self.dataset, value, token_type)

    @final
    def decode(self, token: str, **kwargs: Any) -> str:
        """Decode a Hopwise token string back to a dataset value.

        .. warning::
            This method is **sealed** — subclasses must not override it.

        :param token: The encoded token string.
        :param kwargs: Pass ``real_token=True`` to resolve item tokens
            to their human-readable name.
        :returns: The decoded dataset value.
        :rtype: str
        """
        return hopwise_decode(self.dataset, token, real_token=kwargs.get("real_token"))

    # -- optional endpoints ----------------------------------------------------

    def search(self, request):
        """Search items (no-op in the default wrapper)."""
        return None

    def info(self, request):
        """Return item information (no-op in the default wrapper)."""
        return None
