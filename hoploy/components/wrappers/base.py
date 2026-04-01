"""Base class for hoploy model wrappers.

Plugin authors should subclass
:class:`~hoploy.components.wrappers.default_hopwise_wrapper.DefaultHopwiseWrapper`
and override only the three pipeline hooks described below.

Customisation surface
---------------------

REQUIRED — must be overridden
    :meth:`distill`   Convert an API request into model-ready inputs.
    :meth:`handle`    Configure wrapper state for the current request.
    :meth:`expand`    Decode model output into an API response dict.

OPTIONAL — override to extend
    :meth:`search`    Item search endpoint; returns ``None`` by default.
    :meth:`info`      Item info endpoint; returns ``None`` by default.

Framework internals (``recommend``, ``encode``, ``decode``) are
implemented and sealed by
:class:`~hoploy.components.wrappers.default_hopwise_wrapper.DefaultHopwiseWrapper`
— they do **not** appear here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hoploy.core.config import Config


class BaseWrapper(ABC):
    """Abstract base class for Hopwise model wrappers.

    The three *pipeline hooks* (``distill`` / ``handle`` / ``expand``)
    define the full input → output lifecycle of one recommendation
    request.  Everything else is implemented — and sealed — by
    :class:`~hoploy.components.wrappers.default_hopwise_wrapper.DefaultHopwiseWrapper`.
    """

    # ------------------------------------------------------------------
    # Pipeline hooks — REQUIRED
    # ------------------------------------------------------------------

    @abstractmethod
    def distill(self, request: Config) -> list[str]:
        """Convert an API request into model-ready token strings.

        :param request: A :class:`~hoploy.core.config.Config` wrapping
            the API payload, typically containing an ``input`` key with
            raw item identifiers.
        :returns: A list of Hopwise-encoded input strings that can be
            passed directly to ``recommend()``.
        :rtype: list[str]
        """
        ...

    @abstractmethod
    def handle(self, request: Config) -> BaseWrapper:
        """Configure wrapper state for the current request.

        :param request: A :class:`~hoploy.core.config.Config` wrapping
            the API payload.
        :returns: ``self`` — enables method chaining.
        """
        ...

    @abstractmethod
    def expand(self, values: Any, request: Config) -> dict[str, Any]:
        """Decode model output into a JSON-serialisable response dict.

        :param values: Raw output from ``recommend()`` — typically a
            ``(scores, item_ids, explanations)`` tuple.
        :param request: The original request
            :class:`~hoploy.core.config.Config`.
        :returns: A JSON-serialisable response dictionary.
        :rtype: dict[str, Any]
        """
        ...

    # ------------------------------------------------------------------
    # Optional endpoint hooks — override to extend
    # ------------------------------------------------------------------

    def search(self, request: Config) -> dict[str, Any] | None:
        """Item search endpoint.

        Override to provide a custom search implementation.

        :returns: A response dict, or ``None`` (default).
        """
        return None

    def info(self, request: Config) -> dict[str, Any] | None:
        """Item information endpoint.

        Override to provide item details.

        :returns: A response dict, or ``None`` (default).
        """
        return None
