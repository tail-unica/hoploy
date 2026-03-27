import math

from hopwise.utils import PathLanguageModelingTokenType

from hoploy.model.wrappers.default import DefaultHopwiseWrapper
from hoploy.core.registry import Model
from hoploy.core.utils import hopwise_encode, hopwise_decode, id2tokenizer_token

from hoploy import logger

from .processors import user_sample_compatible_features


# ---- Explanation helpers ----

sensory_features_it: dict[str, str] = {
    "LIGHT": "luminosità",
    "SPACE": "spazi",
    "CROWD": "affollamento",
    "NOISE": "rumore",
    "ODOR": "odori",
}


def _entity_to_italian(entity_name: str) -> str:
    parts = entity_name.split(".")
    if len(parts) >= 2 and parts[0] == "SensoryFeature":
        return sensory_features_it.get(parts[1], entity_name)
    return entity_name


def _entity_target_key(entity_name: str) -> str:
    parts = entity_name.split(".")
    if len(parts) >= 2 and parts[0] == "SensoryFeature":
        return parts[1]
    return entity_name


def _match_force_path(raw_tokens, force_paths, dataset):
    relations = [t for t in raw_tokens if t.startswith(PathLanguageModelingTokenType.RELATION.token)]
    for idx, fp in enumerate(force_paths):
        resolved = []
        for rel_name in fp:
            rk = dataset.ui_relation if rel_name == "[UI-Relation]" else rel_name
            rid = dataset.field2token_id[dataset.relation_field].get(rk)
            if rid is None:
                resolved = []
                break
            resolved.append(PathLanguageModelingTokenType.RELATION.token + str(rid))
        if relations == resolved:
            return idx
    return -1


def _format_explanation(template_config, raw_tokens, real_tokens, better_readability=True):
    def _wrap(t):
        return f"**{t}**" if better_readability else t

    items, entities, users, entity_targets = [], [], [], []
    for raw, real in zip(raw_tokens, real_tokens):
        if raw.startswith(PathLanguageModelingTokenType.ITEM.token):
            items.append(real)
        elif raw.startswith(PathLanguageModelingTokenType.ENTITY.token):
            entity_targets.append(_entity_target_key(real))
            entities.append(_entity_to_italian(real))
        elif raw.startswith(PathLanguageModelingTokenType.USER.token):
            users.append(real)

    if isinstance(template_config, str):
        template = template_config
    else:
        target = template_config.get("target")
        templates = template_config.get("template", {})
        selected = None
        if isinstance(target, str) and len(target) >= 2 and target[1:].isdigit():
            ti = int(target[1:]) - 1
            tt = target[0]
            if tt == "E" and 0 <= ti < len(entity_targets):
                selected = templates.get(entity_targets[ti])
            elif tt == "I" and 0 <= ti < len(items):
                selected = templates.get(items[ti])
        template = selected or next(iter(templates.values()), "")

    result = template
    for i, item in enumerate(items, 1):
        result = result.replace(f"%(I{i})", _wrap(item))
    for i, entity in enumerate(entities, 1):
        result = result.replace(f"%(E{i})", _wrap(entity))
    for i, user in enumerate(users, 1):
        result = result.replace(f"%(U{i})", _wrap(user))
    return result


def _token2real_token(token, dataset):
    """Convert hopwise token to human-readable form, resolving item names."""
    if token.startswith(PathLanguageModelingTokenType.ITEM.token):
        iid = int(token[1:])
        name_field = dataset.field2id_token["name"][dataset.item_feat[iid]["name"]]
        return " ".join(n for n in name_field if n != "[PAD]")
    return hopwise_decode(dataset, token)


# ---- Autism model wrapper ----

@Model("autism_model")
class AutismWrapper(DefaultHopwiseWrapper):
    def __init__(self, cfg):
        super().__init__(cfg)

    def distill(self, **payload):
        """Zero-shot: build inputs from preferences + compatible sensory features."""
        preferences = payload.get("preferences", [])
        aversions = payload.get("aversions")
        if not preferences:
            logger.error("No preferences provided for zero-shot recommendation.")
            return []

        separator = self.dataset.path_token_separator
        bos = self.dataset.tokenizer.bos_token

        # Item preference inputs
        raw_inputs = [
            separator.join([bos, hopwise_encode(self.dataset, pref, PathLanguageModelingTokenType.ITEM.token)])
            for pref in preferences
        ]

        # Compatible sensory feature inputs
        if aversions:
            if isinstance(aversions, list):
                aversions = {a["feature_name"]: a["rating"] for a in aversions}
            for feature in user_sample_compatible_features(aversions):
                try:
                    token = hopwise_encode(self.dataset, feature, PathLanguageModelingTokenType.ENTITY.token)
                    raw_inputs.append(separator.join([bos, token]))
                except KeyError:
                    pass

        logger.debug(f"Autism distill: {len(raw_inputs)} raw inputs")
        return raw_inputs

    def config(self, **payload):
        """Configure generation params; merge preferences into previous_recommendations."""
        super().config(**payload)

        # Extend previous_recommendations with preferences to avoid re-recommending them
        prev = list(payload.get("previous_recommendations", []) or [])
        prev.extend(payload.get("preferences", []))
        if prev:
            token_ids = id2tokenizer_token(self.dataset, prev, "item")
            payload["previous_recommendations"] = token_ids
        return self

    def expand(self, values):
        """Convert raw model output to the autism RecommendationResponse schema."""
        if values is None:
            return None

        scores, recommendations, explanations = values

        # Filter inf scores
        valid = [(s, r, e) for s, r, e in zip(scores, recommendations, explanations) if not math.isinf(float(s))]
        if not valid:
            return None
        scores, recommendations, explanations = zip(*valid)

        dataset = self.dataset
        force_paths = list(getattr(self.cfg, "force_paths", []) or [])
        force_path_explanations = list(getattr(self.cfg, "force_path_explanations", []) or [])
        better_readability = bool(getattr(self.cfg, "better_readability", True))

        result_items = []
        for score, rec_id, raw_exp in zip(scores, recommendations, explanations):
            # Resolve item name
            rec_token = PathLanguageModelingTokenType.ITEM.token + str(rec_id)
            place_name = _token2real_token(rec_token, dataset)

            # Build explanation
            raw_tokens = raw_exp[1:]  # skip BOS
            real_tokens = [_token2real_token(t, dataset) for t in raw_tokens]

            explanation_text = None
            if force_paths and force_path_explanations:
                path_idx = _match_force_path(raw_tokens, force_paths, dataset)
                if 0 <= path_idx < len(force_path_explanations):
                    explanation_text = _format_explanation(
                        force_path_explanations[path_idx], raw_tokens, real_tokens,
                        better_readability=better_readability,
                    )
            if explanation_text is None:
                explanation_text = " ".join(real_tokens)

            result_items.append({
                "place": place_name,
                "score": float(score),
                "explanation": explanation_text,
            })

        return {
            "recommendations": result_items,
        }