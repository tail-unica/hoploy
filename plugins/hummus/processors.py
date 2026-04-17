from hopwise.utils import PathLanguageModelingTokenType

from hoploy.components import (
    DefaultHopwiseLogitsProcessor,
    DefaultHopwiseSequenceScorePostProcessor,
    RestrictedHopwiseLogitsProcessor,
)
from hoploy.registry import LogitsProcessor, SequenceProcessor

from hoploy import logger
from hoploy.core.catalog import get_catalog
from hoploy.core.utils import get_valid_item_ids


@LogitsProcessor("hummus_logits_processor")
class HummusLogitsProcessor(DefaultHopwiseLogitsProcessor):
    """Hummus-specific logits processor.

    Masks previously recommended recipes and user preferences so the
    model does not re-recommend known items.
    """

    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)
        self._catalog = get_catalog(str(cfg.dataset))
        self._valid_ids = get_valid_item_ids(dataset)

    def _names_to_recipe_ids(self, names):
        result = []
        for name in names:
            recipe_id = self._catalog.name_index.get(name.lower())
            if recipe_id is not None and recipe_id in self._valid_ids:
                result.append(recipe_id)
            else:
                resolved = self._catalog.resolve_to_valid(name, self._valid_ids, top_k=1)
                if resolved:
                    result.append(resolved[0])
        return result

    def handle(self, request):
        super().handle(request)
        prev_names = list(getattr(request, "previous_recommendations", None) or [])
        prev_names.extend(getattr(request, "preferences", []))
        recipe_ids = self._names_to_recipe_ids(prev_names)
        if recipe_ids:
            hopwise_ids = []
            for rid in recipe_ids:
                try:
                    hopwise_ids.append(
                        self.encode(rid, PathLanguageModelingTokenType.ITEM.token)
                    )
                except KeyError:
                    logger.warning(f"recipe_id '{rid}' not in model vocabulary, skipping.")
            self.set_previous_recommendations(hopwise_ids or None)
        else:
            self.set_previous_recommendations(None)
        return self


@LogitsProcessor("hummus_restricted_logits_processor")
class HummusRestrictedLogitsProcessor(RestrictedHopwiseLogitsProcessor):
    """Hummus-specific restricted logits processor.

    Accepts hard and soft restriction lists from the request and
    translates recipe/ingredient names to entity tokens.
    """

    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)
        self._catalog = get_catalog(str(cfg.dataset))
        self._valid_ids = get_valid_item_ids(dataset)

    def _names_to_recipe_ids(self, names):
        result = []
        for name in names:
            recipe_id = self._catalog.name_index.get(name.lower())
            if recipe_id is not None and recipe_id in self._valid_ids:
                result.append(recipe_id)
            else:
                resolved = self._catalog.resolve_to_valid(name, self._valid_ids, top_k=1)
                if resolved:
                    result.append(resolved[0])
        return result

    def handle(self, request):
        self.clear_restrictions()
        hard = getattr(request, "hard_restrictions", None)
        soft = getattr(request, "soft_restrictions", None)
        if not hard and not soft:
            return self

        hard_ids = []
        if hard:
            for rid in self._names_to_recipe_ids(hard):
                try:
                    hard_ids.append(
                        self.encode(rid, PathLanguageModelingTokenType.ITEM.token)
                    )
                except KeyError:
                    pass

        soft_ids = []
        if soft:
            for rid in self._names_to_recipe_ids(soft):
                try:
                    soft_ids.append(
                        self.encode(rid, PathLanguageModelingTokenType.ITEM.token)
                    )
                except KeyError:
                    pass

        if hard_ids or soft_ids:
            self.set_restrictions(
                hard_restrictions=hard_ids or None,
                soft_restrictions=soft_ids or None,
            )
            logger.info(
                f"Hummus restricted processor: {len(hard_ids)} hard, {len(soft_ids)} soft restrictions"
            )
        return self


@SequenceProcessor("hummus_sequence_processor")
class HummusSequenceProcessor(DefaultHopwiseSequenceScorePostProcessor):
    """Hummus-specific sequence score post-processor.

    Identical to the default for now — placeholder for future
    domain-specific scoring adjustments.
    """

    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)
