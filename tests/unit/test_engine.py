from __future__ import annotations

from hoploy.core.state import GenerationState


class TestGenerationState:
    def test_step_forward(self) -> None:
        state = GenerationState(
            user_id="u1",
            step=0,
            input_ids=(1, 2, 3),
            generated_ids=(),
            constraint_config={},
        )
        next_state = state.step_forward(42, score=0.9)

        assert next_state.step == 1
        assert next_state.generated_ids == (42,)
        assert next_state.context["score"] == 0.9
        # original is unchanged (frozen)
        assert state.step == 0
        assert state.generated_ids == ()

    def test_multiple_steps(self) -> None:
        state = GenerationState(
            user_id="u2",
            step=0,
            input_ids=(10,),
            generated_ids=(),
            constraint_config={},
        )
        for tok in [100, 200, 300]:
            state = state.step_forward(tok)
        assert state.step == 3
        assert state.generated_ids == (100, 200, 300)
