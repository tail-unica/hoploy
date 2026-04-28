import numpy as np

from hoploy.components import DefaultHopwiseSequenceScorePostProcessor, ForcedLogitsProcessor, ForcedSequenceScorePostProcessor, RestrictedHopwiseLogitsProcessor
from hoploy.registry import SequenceProcessor, LogitsProcessor

from hopwise.utils import PathLanguageModelingTokenType

from hoploy import logger
from hoploy.core.catalog import get_catalog

# ---- Sensory compatibility helpers ----

sensory_features_map: dict[str, list[str]] = {
    "LIGHT": ["bright_light", "dim_light"],
    "SPACE": ["wide_space", "narrow_space"],
    "CROWD": ["crowd"],
    "NOISE": ["noise"],
    "ODOR": ["odor"],
}


def user_feature_compatibility(aversions: dict[str, float], features: dict[str, float]) -> dict[str, bool]:
    """Determine per-feature sensory compatibility for a user.

    :param aversions: User's sensory aversion ratings keyed by sub-feature.
    :type aversions: dict[str, float]
    :param features: Item's sensory feature values keyed by feature name.
    :type features: dict[str, float]
    :returns: Map of feature name to compatibility boolean.
    :rtype: dict[str, bool]
    """
    INDIVIDUAL_COMPATIBILITY_THRESHOLD = 3

    def _aversion_high(ft_value, ua):
        return 1 + (ua - 1) * (ft_value - 1) / (5 - 1)

    def _aversion_low(ft_value, ua):
        return 1 + (ft_value - 5) * (1 - ua) / (5 - 1)

    result = {}
    for feature, aversions_list in sensory_features_map.items():
        if len(aversions_list) == 2:
            low = aversions.get(aversions_list[0], 1.0)
            high = aversions.get(aversions_list[1], 1.0)
            result[feature] = (
                6 - max(_aversion_low(features[feature], low), _aversion_high(features[feature], high))
                > INDIVIDUAL_COMPATIBILITY_THRESHOLD
            )
        else:
            av = aversions.get(aversions_list[0], 1.0)
            result[feature] = 6 - _aversion_high(features[feature], av) > INDIVIDUAL_COMPATIBILITY_THRESHOLD
    return result


def user_feature_mask(aversions: dict[str, float]) -> list[str]:
    """Return non-compatible sensory features as entity names.

    Generates a dense Likert grid (step 0.1) and collects every value
    that is incompatible with the user's aversions.  Entity names that
    do not exist in the loaded dataset are silently skipped at encoding
    time; this function intentionally over-generates to remain dataset-
    agnostic.

    :param aversions: User's sensory aversion ratings.
    :type aversions: dict[str, float]
    :returns: Entity names, e.g. ``'SensoryFeature.NOISE.2.3'``.
    :rtype: list[str]
    """
    LIKERT_STEP = 0.1
    LIKERT_RANGE = np.arange(1.0, 5.0 + LIKERT_STEP, LIKERT_STEP)
    non_compatible = set()
    for fv in LIKERT_RANGE:
        compat = user_feature_compatibility(aversions, {f: fv for f in sensory_features_map})
        for feature, ok in compat.items():
            if not ok:
                non_compatible.add(f"SensoryFeature.{feature}.{fv:.1f}")
    return list(non_compatible)


def user_sample_compatible_features(aversions: dict[str, float]) -> list[str]:
    """Sample one compatible sensory value per feature.

    Identifies all compatible values on the Likert grid, then selects
    one value from near the centre of the largest contiguous compatible
    band (circular gap logic to avoid boundary effects).

    :param aversions: User's sensory aversion ratings.
    :type aversions: dict[str, float]
    :returns: Entity names, e.g. ``'SensoryFeature.LIGHT.3.0'``.
    :rtype: list[str]
    """
    LIKERT_STEP = 0.1
    LIKERT_RANGE = np.arange(1.0, 5.0 + LIKERT_STEP, LIKERT_STEP)

    compatible: dict[str, list[float]] = {}
    for val in LIKERT_RANGE:
        val = round(val, 1)
        compat = user_feature_compatibility(aversions, {f: val for f in sensory_features_map})
        for feature, ok in compat.items():
            if ok:
                compatible.setdefault(feature, []).append(val)

    sampled = []
    for feature, vals in compatible.items():
        if not vals:
            continue

        arr = np.array(vals)

        # -- Circular gap logic --
        # Find the largest gap in the compatible range (including the
        # wrap-around gap between the last and first values across 5→1)
        # and rotate the array so that gap becomes the boundary.  This
        # ensures the mid-point selection always picks from the widest
        # contiguous band rather than an arbitrary split.
        diffs = np.diff(arr)
        wrap_gap = (arr[0] - 1.0) + (5.0 - arr[-1])
        all_gaps = np.append(diffs, wrap_gap)
        max_gap_idx = np.argmax(all_gaps)
        if max_gap_idx != len(all_gaps) - 1:
            arr = np.roll(arr, -(max_gap_idx + 1))

        n = len(arr)
        mid = n // 2
        jitter = max(1, int(n * 0.1))
        idx = int(np.clip(mid + np.random.randint(-jitter, jitter + 1), 0, n - 1))
        sampled.append(f"SensoryFeature.{feature}.{arr[idx]:.1f}")
    return sampled


# ---- Plugin processors ----

@SequenceProcessor("autism_sequence_processor")
class AutismSequenceProcessor(ForcedSequenceScorePostProcessor):
    """Autism-specific sequence score post-processor.

    Extends :class:`~hoploy.components.processors.forced_sequence_processor.ForcedSequenceScorePostProcessor`
    to apply forced relation-path filtering and per-type diversity boosting
    as configured in the plugin's ``sequence_processor`` config section.
    """

    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)


@LogitsProcessor("autism_logits_processor")
class AutismLogitsProcessor(ForcedLogitsProcessor):
    """Autism-specific logits processor.

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


@LogitsProcessor("autism_restricted_logits_processor")
class AutismRestrictedLogitsProcessor(RestrictedHopwiseLogitsProcessor):
    """Autism-specific restricted logits processor.

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
        :rtype: AutismRestrictedLogitsProcessor
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
                logger.info(f"Autism restricted processor: {len(hopwise_ids)} hard restrictions set")

        return self