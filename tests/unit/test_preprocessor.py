"""Unit tests for DefaultPreprocessor."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from hoploy.preprocessing.default import DefaultPreprocessor


class _FakeTokenizer:
    """Minimal tokenizer mock."""

    all_special_ids = [0, 1, 2]  # BOS, EOS, PAD

    def __call__(self, texts, *, return_tensors="pt", add_special_tokens=False):
        # Produce a simple encoded result: one id per char (just for testing)
        ids = []
        for t in texts:
            row = [ord(c) % 100 + 10 for c in t]  # avoids special ids 0-2
            ids.append(row)
        max_len = max(len(r) for r in ids) if ids else 0
        for r in ids:
            r.extend([0] * (max_len - len(r)))  # pad with 0
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


def _make_request(raw_inputs: list[str], **kwargs) -> dict:
    return {"raw_inputs": raw_inputs, "user_id": "u1", "device": "cpu", **kwargs}


class TestDefaultPreprocessor:
    def test_basic_tokenisation(self):
        prep = DefaultPreprocessor()
        tok = _FakeTokenizer()
        req = _make_request(["hello"])
        input_ids, state = prep.run(req, tok)

        assert isinstance(input_ids, torch.Tensor)
        assert state.user_id == "u1"
        assert state.step == 0
        assert len(state.generated_ids) == 0

    def test_all_special_tokens_dropped(self):
        """When every non-BOS token is special, the result should be empty."""
        prep = DefaultPreprocessor()
        tok = _FakeTokenizer()
        # Create input that will be entirely special after BOS
        tok_mock = MagicMock(wraps=tok)
        # Return tensor of all special ids
        tok_mock.__call__ = MagicMock(return_value={
            "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        })
        tok_mock.all_special_ids = [0, 1, 2]

        req = _make_request(["x"])
        input_ids, state = prep.run(req, tok_mock)
        assert input_ids.shape[0] == 0

    def test_state_carries_extra_keys(self):
        prep = DefaultPreprocessor()
        tok = _FakeTokenizer()
        req = _make_request(["abc"], extra_key="value")
        _, state = prep.run(req, tok)
        assert state.context["extra_key"] == "value"
