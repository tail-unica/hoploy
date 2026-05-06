from hopwise.utils import PathLanguageModelingTokenType

from hoploy.components import ForcedLogitsProcessor, RestrictedHopwiseLogitsProcessor
from hoploy.registry import LogitsProcessor

from hoploy import logger
from hoploy.core.catalog import get_catalog

from .sensory import user_feature_mask


@LogitsProcessor("poi_logits_processor")
class PointsOfInterestLogitsProcessor(ForcedLogitsProcessor):
    """Points of Interest-specific logits processor.

    Extends the forced-path processor to merge user preferences into
    previous recommendations, preventing the model from re-recommending
    already-known places.
    """
    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)
        self._catalog = get_catalog()

    def _names_to_poi_ids(self, names):
        """Convert place names to dataset POI ids.

        :param names: Human-readable place names.
        :type names: list[str]
        :returns: Matching POI ids.
        :rtype: list[str]
        """
        result = []
        for name in names:
            poi_id = self._catalog.name_index.get(name.lower())
            if poi_id is not None:
                result.append(poi_id)
        return result

    def handle(self, request):
        """Set previous recommendations from the request.

        Preferences are merged in so already-known places are excluded.
        Force paths are handled by the parent.

        :param request: Incoming recommendation request.
        :type request: Config
        :returns: Self.
        :rtype: AutismLogitsProcessor
        """
        super().handle(request)
        prev_names = list(getattr(request, "previous_recommendations", None) or [])
        prev_names.extend(request.preferences)
        poi_ids = self._names_to_poi_ids(prev_names)
        if poi_ids:
            hopwise_ids = [
                self.encode(pid, PathLanguageModelingTokenType.ITEM.token)
                for pid in poi_ids
            ]
            self.set_previous_recommendations(hopwise_ids)
        else:
            self.set_previous_recommendations(None)
        return self


@LogitsProcessor("poi_restricted_logits_processor")
class PointsOfInterestRestrictedLogitsProcessor(RestrictedHopwiseLogitsProcessor):
    """Points of Interest-specific restricted logits processor.

    Computes hard token restrictions from the user's sensory aversions
    so the model cannot generate incompatible features.
    """
    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)

    def handle(self, request):
        """Compute hard restrictions from the user's aversions.

        :param request: Incoming recommendation request.
        :type request: Config
        :returns: Self.
        :rtype: PointsOfInterestRestrictedLogitsProcessor
        """
        self.clear_restrictions()
        aversions = getattr(request, "aversions", None)
        if not aversions:
            return self

        # aversions come from the schema as a list of dicts [{feature_name, rating}, ...]
        if not isinstance(aversions, dict):
            aversions = {a["feature_name"]: a["rating"] for a in aversions}

        hard_features = user_feature_mask(aversions)
        if hard_features:
            hopwise_ids = []
            for feat in hard_features:
                try:
                    hopwise_ids.append(
                        self.encode(feat, PathLanguageModelingTokenType.ENTITY.token)
                    )
                except KeyError:
                    pass  # entity not present in this dataset, skip
            if hopwise_ids:
                self.set_restrictions(hard_restrictions=hopwise_ids)
                logger.info(f"Points of Interest restricted processor: {len(hopwise_ids)} hard restrictions set")

        return self
