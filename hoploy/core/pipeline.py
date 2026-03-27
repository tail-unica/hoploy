from .registry import PluginRegistry, load_plugin

from hoploy import logger

class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self._ready = False

        for plugin_cfg in self.cfg.plugin.raw.values():
            load_plugin(plugin_cfg)
        
        # Model initialization
        logger.info("Initializing model")
        self.model_name, model_cfg = list(self.cfg.model.raw.items())[0]
        self.model = PluginRegistry.get(model_cfg.name)(model_cfg)

        # Logits processor initialization
        logger.info("Initializing logits processors")
        self.logits_processors = []
        for name, processor_cfg in self.cfg.logits_processors.raw.items():
            processor = PluginRegistry.get(processor_cfg.name)(
                dataset=self.model.dataset,
                cfg=processor_cfg,
            )
            self.logits_processors.append((name, processor))

        # Sequence processor initialization
        logger.info("Initializing sequence processor")
        sequence_items = list(self.cfg.sequence_processor.raw.items())
        if sequence_items:
            self.sequence_processor_name, sequence_cfg = sequence_items[0]
            self.sequence_processor = PluginRegistry.get(sequence_cfg.name)(
                dataset=self.model.dataset,
                cfg=sequence_cfg,
            )
        else:
            self.sequence_processor = None

        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    async def shutdown(self):
        self._ready = False

    def run(self, **payload):
        """
        Run the pipeline with the given input arguments.

        :param payload: API request parameters
        :type payload: dict
        """
        inputs = self.model.distill(**payload)

        # NOTE: cfg.update creates a copy of the config with updated values, so we can safely call it multiple times without mutating the original config
        model_kwargs = self.cfg.model.update(**{self.model_name: payload}).to_dict().get(self.model_name, {})
        
        sequence_processor_config = None
        if self.sequence_processor is not None:
            sequence_kwargs = self.cfg.sequence_processor.update(**{self.sequence_processor_name: payload}).to_dict().get(
                self.sequence_processor_name,
                {},
            )
            sequence_processor_config = self.sequence_processor.config(**sequence_kwargs)

        self.model.config(**model_kwargs)
        self.model.update_processors(
            logits_processors=[
                processor.config(**self.cfg.logits_processors.update(**{name: payload}).to_dict().get(name, {}))
                for name, processor in self.logits_processors
            ],
            sequence_processor=sequence_processor_config
        )
        
        out = self.model.recommend(inputs)
        logger.debug(f"Model output: {out}")

        return self.model.expand(out)
    
    async def info(self, **kwargs):
        return None

    async def search(self, **kwargs):
        return None