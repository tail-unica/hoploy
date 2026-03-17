from __future__ import annotations

from hoploy.postprocessing.default import DefaultSequencePostprocessor


class TestDefaultSequencePostprocessor:
    def test_sanitize_scores(self) -> None:
        scores = [1.0, float("inf"), float("-inf"), float("nan"), 0.5]
        clean = DefaultSequencePostprocessor.sanitize_scores(scores)
        assert clean == [1.0, 0.0, 0.0, 0.0, 0.5]

    def test_select_top_k(self) -> None:
        pp = DefaultSequencePostprocessor()
        seqs = [("a", "x", 0.1), ("b", "y", 0.9), ("c", "z", 0.5)]
        top = pp.select_top_k(seqs, k=2)
        assert len(top) == 2
        assert top[0][2] == 0.9
