from __future__ import annotations

import torch

from hoploy.core.state import GenerationState
from hoploy.logit_processors.builtins.temperature import TemperatureLogitProcessor
from hoploy.logit_processors.builtins.top_k import TopKLogitProcessor
from hoploy.logit_processors.chain import LogitProcessorChain


def _dummy_state() -> GenerationState:
    return GenerationState(
        user_id="u1",
        step=0,
        input_ids=(1,),
        generated_ids=(),
        constraint_config={},
    )


class TestTemperatureProcessor:
    def test_scales_logits(self) -> None:
        proc = TemperatureLogitProcessor(temperature=2.0)
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = proc(logits, _dummy_state())
        assert torch.allclose(result, logits / 2.0)


class TestTopKProcessor:
    def test_masks_outside_top_k(self) -> None:
        proc = TopKLogitProcessor(k=2)
        logits = torch.tensor([[1.0, 3.0, 2.0]])
        result = proc(logits, _dummy_state())
        # Only top-2 (indices 1, 2) should survive
        assert result[0, 0] == float("-inf")
        assert result[0, 1] == 3.0
        assert result[0, 2] == 2.0


class TestLogitProcessorChain:
    def test_chain_applies_in_order(self) -> None:
        chain = LogitProcessorChain([
            TemperatureLogitProcessor(temperature=2.0),
            TopKLogitProcessor(k=2),
        ])
        logits = torch.tensor([[1.0, 3.0, 2.0]])
        result = chain(logits, _dummy_state())
        assert result[0, 0] == float("-inf")
