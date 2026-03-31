import logging

from .registry import PluginRegistry, load_plugin

logger = logging.getLogger(__name__)

class Pipe:
    """Request-processing pipeline.

    Loads plugins, instantiates the wrapper and all processors at
    construction time, then orchestrates them on each call to :meth:`run`.

    :param cfg: The full application :class:`~hoploy.core.config.Config`.
    :type cfg: Config
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._ready = False

        for plugin_cfg in self.cfg.plugin.raw.values():
            load_plugin(plugin_cfg)
        
        # Wrapper initialization
        logger.info("Initializing wrapper")
        wrapper_cfg = next(iter(self.cfg.wrapper))
        self.wrapper = PluginRegistry.get(wrapper_cfg.name)(wrapper_cfg)

        # Logits processor initialization
        logger.info("Initializing logits processors")
        self.logits_processors = []
        for processor_cfg in self.cfg.logits_processors:
            processor = PluginRegistry.get(processor_cfg.name)(
                dataset=self.wrapper.dataset,
                cfg=processor_cfg,
            )
            self.logits_processors.append(processor)

        # Sequence processor initialization
        logger.info("Initializing sequence processor")
        self.sequence_processor = next(iter(self.cfg.sequence_processor), None)
        if self.sequence_processor is not None:
            sequence_cfg = self.sequence_processor
            self.sequence_processor = PluginRegistry.get(sequence_cfg.name)(
                dataset=self.wrapper.dataset,
                cfg=sequence_cfg,
            )

        self._ready = True

    def is_ready(self) -> bool:
        """Return ``True`` when the pipeline is fully initialised."""
        return self._ready

    async def shutdown(self):
        """Mark the pipeline as unavailable."""
        self._ready = False

    def run(self, request):
        """Execute the full recommendation pipeline.

        :param request: A :class:`~hoploy.core.config.Config` wrapping the
            API request payload.
        :type request: Config
        :returns: The enriched response dict produced by
            :meth:`~hoploy.components.wrappers.base.BaseWrapper.expand`.
        :rtype: dict
        """
        inputs = self.wrapper.distill(request)

        self.wrapper.handle(request)
        self.wrapper.update_processors(
            logits_processors=[
                processor.handle(request)
                for processor in self.logits_processors
            ],
            sequence_processor=(
                self.sequence_processor.handle(request)
                if self.sequence_processor is not None else None
            ),
        )
        
        out = self.wrapper.recommend(inputs)
        logger.debug(f"Wrapper output: {out}")

        return self.wrapper.expand(out, request)