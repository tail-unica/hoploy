from abc import ABC, abstractmethod


class BaseWrapper(ABC):
    """Abstract base class for model wrappers.

    Plugin authors should subclass
    :class:`~hoploy.components.wrappers.default_hopwise_wrapper.DefaultHopwiseWrapper`
    and override only the three hooks: :meth:`distill`, :meth:`handle`,
    and :meth:`expand`.

    Internal methods (:meth:`recommend`, :meth:`encode`, :meth:`decode`)
    are implemented by the default wrapper and should **not** be overridden.
    """

    @abstractmethod
    def distill(self, request):
        """Convert an API request into model input tokens.

        :param request: A :class:`~hoploy.core.config.Config` wrapping the
            API payload.
        :type request: Config
        """
        ...

    @abstractmethod
    def handle(self, request):
        """Configure the wrapper for the current request.

        :param request: A :class:`~hoploy.core.config.Config` wrapping the
            API payload.
        :type request: Config
        """
        ...

    @abstractmethod
    def expand(self, values, request) -> dict:
        """Convert model output into an API response dict.

        :param values: The raw output from :meth:`recommend`.
        :param request: The original request :class:`~hoploy.core.config.Config`.
        :type request: Config
        :returns: A JSON-serialisable response dict.
        :rtype: dict
        """
        ...

    def recommend(self, inputs):
        """Generate recommendations from tokenized inputs.

        :param inputs: Hopwise-tokenized input strings.
        :type inputs: list[str]
        """
        raise NotImplementedError

    def encode(self, value, token_type):
        """Encode a dataset value to a Hopwise token string.

        :param value: The raw dataset value.
        :param token_type: Token prefix string.
        :type token_type: str
        """
        raise NotImplementedError

    def decode(self, token, **kwargs):
        """Decode a Hopwise token string back to a dataset value.

        :param token: The encoded token string.
        :type token: str
        """
        raise NotImplementedError

    def info(self, request):
        """Return information about a specific item.

        :param request: Item identifier or :class:`Config` with an
            ``item`` key.
        :returns: An info dict, or ``None`` if not found.
        :rtype: dict | None
        """
        return None
