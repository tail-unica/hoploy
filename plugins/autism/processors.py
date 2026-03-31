import numpy as np
from numpy import arange

from hoploy.components import DefaultHopwiseSequenceScorePostProcessor, DefaultHopwiseLogitsProcessor, DefaultRestrictedHopwiseLogitsProcessor
from hoploy.registry import SequenceProcessor, LogitsProcessor
from hoploy.core.utils import hopwise_encode, id2tokenizer_token

from hoploy import logger

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
        return 1 + (ua - 1) * (ft_value - 1) / 4

    def _aversion_low(ft_value, ua):
        return 1 + (ft_value - 5) * (1 - ua) / 4

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

    Iterates over the full Likert range and collects every feature
    value that is incompatible with the user's aversions.

    :param aversions: User's sensory aversion ratings.
    :type aversions: dict[str, float]
    :returns: Entity names, e.g. ``'SensoryFeature.NOISE.2.3'``.
    :rtype: list[str]
    """
    LIKERT_STEP = 0.1
    LIKERT_RANGE = arange(1.0, 5.0 + LIKERT_STEP, LIKERT_STEP)
    non_compatible = set()
    for fv in LIKERT_RANGE:
        compat = user_feature_compatibility(aversions, {f: fv for f in sensory_features_map})
        for feature, ok in compat.items():
            if not ok:
                non_compatible.add(f"SensoryFeature.{feature}.{fv:.1f}")
    return list(non_compatible)


def user_sample_compatible_features(aversions: dict[str, float]) -> list[str]:
    """Sample one compatible sensory value per feature.

    Values are biased towards the middle of the compatible range.

    :param aversions: User's sensory aversion ratings.
    :type aversions: dict[str, float]
    :returns: Entity names, e.g. ``'SensoryFeature.LIGHT.3.0'``.
    :rtype: list[str]
    """
    LIKERT_STEP = 0.1
    LIKERT_RANGE = np.arange(1.0, 5.0 + LIKERT_STEP, LIKERT_STEP)

    compatible = {}
    for val in LIKERT_RANGE:
        val = round(val, 1)
        ctx = {f: val for f in sensory_features_map}
        compat = user_feature_compatibility(aversions, ctx)
        for feature, ok in compat.items():
            if ok:
                compatible.setdefault(feature, []).append(val)

    sampled = []
    for feature, vals in compatible.items():
        if not vals:
            continue
        arr = np.array(vals)
        n = len(arr)
        mid = n // 2
        jitter = max(1, int(n * 0.1))
        idx = np.clip(mid + np.random.randint(-jitter, jitter + 1), 0, n - 1)
        sampled.append(f"SensoryFeature.{feature}.{arr[idx]:.1f}")
    return sampled


# ---- Plugin processors ----

@SequenceProcessor("autism_sequence_processor")
class AutismSequenceProcessor(DefaultHopwiseSequenceScorePostProcessor):
    """Autism-specific sequence score post-processor."""
    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)


@LogitsProcessor("autism_logits_processor")
class AutismLogitsProcessor(DefaultHopwiseLogitsProcessor):
    """Autism-specific logits processor.

    Extends the default processor to merge user preferences into
    previous recommendations, preventing the model from re-recommending
    already-known places.
    """
    def __init__(self, dataset, cfg, **kwargs):
        super().__init__(dataset, cfg, **kwargs)

    def handle(self, request):
        """Set previous recommendations from the request.

        Preferences are merged in so already-known places are excluded.

        :param request: Incoming recommendation request.
        :type request: Config
        :returns: Self.
        :rtype: AutismLogitsProcessor
        """
        prev_names = list(getattr(request, "previous_recommendations", None) or [])
        prev_names.extend(request.preferences)
        if prev_names:
            token_ids = id2tokenizer_token(self.dataset, prev_names, "item")
            self.set_previous_recommendations(token_ids)
        else:
            self.set_previous_recommendations(None)
        return self


@LogitsProcessor("autism_restricted_logits_processor")
class AutismRestrictedLogitsProcessor(DefaultRestrictedHopwiseLogitsProcessor):
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
            hard_ids = id2tokenizer_token(self.dataset, hard_features, "entity")
            if hard_ids:
                self.set_restrictions(hard_restrictions=hard_ids)
                logger.info(f"Autism restricted processor: {len(hard_ids)} hard restrictions set")

        return self