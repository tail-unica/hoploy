"""Base classes for hoploy logits and sequence processors.

Customisation surface exposed to plugin authors
-----------------------------------------------

``BaseLogitsProcessor`` defines **two levels of hook**:

Level 1 — request wiring  (REQUIRED)
    ``handle(request) -> Self``
    Translate an API request into processor state before generation.
    Every concrete processor MUST implement this.

Level 2 — score adjustment  (optional)
    ``score_adjustment(hopwise_current, hopwise_candidates) -> dict[str, float]``
    Receive the graph-validated candidates for the current generation
    step, expressed as Hopwise IDs (``"type:value"``), and return
    score deltas.  Use ``float('-inf')`` for hard banning.

    No tokenizer knowledge required — the framework maps Hopwise IDs
    to internal token IDs transparently.

For deeper customisation (e.g. forced relation paths), subclasses may
override ``process_scores_rec`` directly; this is an advanced escape
hatch, not a formal hook.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hoploy.core.config import Config


class BaseLogitsProcessor(ABC):
    """Abstract base class for Hopwise logits processors.

    Plugin authors should subclass
    :class:`~hoploy.components.processors.default_logits_processor.DefaultHopwiseLogitsProcessor`
    (or one of its descendants) and override the hook(s) they need.
    See module docstring for the two-level contract.

    The tokenizer and all HuggingFace internals are **never exposed**
    through these hooks; every argument uses Hopwise-level IDs.
    """

    # ------------------------------------------------------------------
    # Level 1 — REQUIRED
    # ------------------------------------------------------------------

    @abstractmethod
    def handle(self, request: Config) -> BaseLogitsProcessor:
        """Configure the processor state for the current request.

        Called once per inference request, before ``__call__`` is invoked
        by the generation loop.

        :param request: A :class:`~hoploy.core.config.Config` wrapping
            the API payload.
        :returns: ``self`` — enables method chaining.

        .. note::
            This is the **only** method every subclass MUST implement.
        """
        ...

    # ------------------------------------------------------------------
    # Level 2 — OPTIONAL hook (override to adjust logit scores)
    # ------------------------------------------------------------------

    def score_adjustment(
        self,
        hopwise_current: str,
        hopwise_candidates: list[str],
    ) -> dict[str, float]:
        """Apply custom score adjustments during generation.

        Called for each unique beam state after graph-constraint
        filtering.  Override in subclasses to implement custom scoring
        logic using Hopwise-level identifiers (never raw tokenizer IDs).

        Identifiers use the format ``"type:value"`` where *type* is one
        of ``item``, ``entity``, ``relation``, ``user``.  For example:
        ``"entity:SensoryFeature.NOISE.2.3"`` or ``"item:55"``.

        :param hopwise_current: Decoded Hopwise ID of the current token.
        :param hopwise_candidates: Decoded Hopwise IDs of the
            graph-valid candidate tokens.
        :returns: Dict of ``{hopwise_id: score_delta}``.  Use
            ``float('-inf')`` to hard-ban a candidate, negative values
            to penalise, positive to boost.  Candidates not in the dict
            keep their original score.
        :rtype: dict[str, float]

        Default behaviour: returns ``{}`` — no adjustment.
        """
        return {}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable identifier (class name by default)."""
        return self.__class__.__name__


class BaseSequenceProcessor(ABC):
    """Abstract base class for sequence score post-processors.

    Plugin authors should subclass
    :class:`~hoploy.components.processors.default_sequence_processor.DefaultHopwiseSequenceScorePostProcessor`
    and override :meth:`handle`.
    """

    @abstractmethod
    def handle(self, request: Config) -> BaseSequenceProcessor:
        """Configure the processor for the current request.

        :param request: A :class:`~hoploy.core.config.Config` wrapping
            the API payload.
        :returns: ``self`` — enables method chaining.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable identifier (class name by default)."""
        return self.__class__.__name__
