import numpy as np


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
