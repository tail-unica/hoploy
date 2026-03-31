from abc import ABC, abstractmethod


class BaseLogitsProcessor(ABC):
    """Abstract base class for logits processors.

    Plugin authors should subclass one of the default processors and
    override :meth:`handle` to translate the API request into processor
    state.

    Available state-setting methods on the default implementations:

    * :meth:`set_previous_recommendations` — mask already-recommended items.
    * :meth:`set_restrictions` — entity / item restrictions.
    * :meth:`clear_restrictions` — reset all restrictions.
    """

    @abstractmethod
    def handle(self, request):
        """Configure the processor for the current request.

        Args:
            request: A ``Config`` object wrapping the API payload.
        """
        ...

    @property
    def name(self):
        """Return the class name as a human-readable identifier."""
        return self.__class__.__name__


class BaseSequenceProcessor(ABC):
    """Abstract base class for sequence post-processors.

    Plugin authors should subclass
    :class:`~hoploy.components.processors.default_sequence_processor.DefaultHopwiseSequenceScorePostProcessor`
    and override :meth:`handle` to adjust scoring / filtering from the
    API payload.
    """

    @abstractmethod
    def handle(self, request):
        """Configure the processor for the current request.

        :param request: A :class:`~hoploy.core.config.Config` wrapping the
            API payload.
        :type request: Config
        """
        ...

    @property
    def name(self):
        """Return the class name as a human-readable identifier."""
        return self.__class__.__name__
