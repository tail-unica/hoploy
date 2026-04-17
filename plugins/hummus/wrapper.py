import math

from hopwise.utils import PathLanguageModelingTokenType

from hoploy.components import DefaultHopwiseWrapper
from hoploy.registry import Wrapper

from hoploy import logger
from hoploy.core.catalog import get_catalog
from hoploy.core.utils import get_valid_item_ids


# --- Catalog helpers ---

_NUTRITIONAL_KEYS = [
    "protein [g]",
    "calories [cal]",
    "caloriesFromFat [cal]",
    "totalFat [g]",
    "saturatedFat [g]",
    "cholesterol [mg]",
    "sodium [mg]",
    "totalCarbohydrate [g]",
    "dietaryFiber [g]",
    "sugars [g]",
]

_GROUP_KEYS = [
    "calories_group",
    "total_fat_group",
    "saturated_fat_group",
    "cholesterol_group",
    "sodium_group",
    "total_carbohydrate_group",
    "dietary_fiber_group",
    "sugars_group",
    "protein_group",
]


def _parse_list_field(value: str) -> list[str]:
    """Parse a bracketed comma-separated list string from the dataset."""
    if not value or value.strip() in ("", "[]"):
        return []
    stripped = value.strip().lstrip("[").rstrip("]")
    return [item.strip() for item in stripped.split(", ") if item.strip()]


def _safe_float(value: str):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _item_to_info(record: dict) -> dict:
    """Convert a catalog record into an InfoResponse-compatible dict."""
    name = record.get("name", "").strip()
    food_item_type = record.get("food_item_type", "recipe").strip()

    # Healthiness
    nutri_group = record.get("nutri_group", "").strip()
    healthiness = {"score": nutri_group} if nutri_group else None

    # Sustainability
    sust_group = record.get("sustainability_score_group", "").strip()
    sustainability = None
    if sust_group:
        sustainability = {
            "score": sust_group,
            "CF": _safe_float(record.get("CF", "")),
            "WF": _safe_float(record.get("WF", "")),
        }

    # Nutritional values
    nutritional_values = {k: _safe_float(record.get(k, "")) for k in _NUTRITIONAL_KEYS}

    # Nutritional value groups
    nutritional_value_groups = {}
    for key in _GROUP_KEYS:
        val = record.get(key, "").strip()
        if val:
            nutritional_value_groups[key.replace("_group", "")] = val

    # Ingredients
    ingredients_list = _parse_list_field(record.get("parsed_ingredients", ""))
    quantities_list = _parse_list_field(record.get("quantities", ""))
    ingredients = None
    if ingredients_list:
        ingredients = {
            "ingredients": ingredients_list,
            "quantities": quantities_list or None,
        }

    return {
        "food_item": name,
        "food_item_type": food_item_type,
        "healthiness": healthiness,
        "sustainability": sustainability,
        "nutritional_values": nutritional_values,
        "nutritional_value_groups": nutritional_value_groups or None,
        "ingredients": ingredients,
        "food_item_url": record.get("recipe_url", "").strip() or None,
    }


# --- Hummus model wrapper ---


@Wrapper("hummus_model")
class HummusWrapper(DefaultHopwiseWrapper):
    """Food recommendation wrapper for the hummus dataset.

    Extends the default wrapper with a recipe catalog and nutritional
    metadata lookup.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._catalog = get_catalog(cfg.dataset)
        self._valid_ids = get_valid_item_ids(self.dataset)
        logger.info(f"Loaded item catalog: {len(self._catalog.items)} items")

    def _names_to_recipe_ids(self, names: list) -> list:
        result = []
        for name in names:
            recipe_id = self._catalog.name_index.get(name.lower())
            if recipe_id is not None and recipe_id in self._valid_ids:
                result.append(recipe_id)
            else:
                resolved = self._catalog.resolve_to_valid(name, self._valid_ids, top_k=1)
                if resolved:
                    logger.info(f"Resolved '{name}' to '{resolved[0]}' via fuzzy match.")
                    result.append(resolved[0])
                else:
                    logger.warning(f"Recipe '{name}' not found in catalog or model vocab, skipping.")
        return result

    # -- pipeline hooks --------------------------------------------------------

    def distill(self, request):
        """Convert preferred recipe names into BOS-prefixed token strings."""
        preferences = request.preferences
        if not preferences:
            logger.error("No preferences provided.")
            return []

        separator = self.dataset.path_token_separator
        bos = self.dataset.tokenizer.bos_token

        raw_inputs = []
        for recipe_id in self._names_to_recipe_ids(preferences):
            try:
                token = self.encode(recipe_id, PathLanguageModelingTokenType.ITEM.token)
                raw_inputs.append(separator.join([bos, token]))
            except KeyError:
                logger.warning(f"recipe_id '{recipe_id}' not found in hopwise dataset, skipping.")

        logger.debug(f"Hummus distill: {len(raw_inputs)} raw inputs")
        return raw_inputs

    def handle(self, request):
        super().handle(request)
        return self

    def expand(self, values, request):
        """Decode model output into a food recommendation response dict."""
        if values is None:
            return None

        scores, recommendations, explanations = values

        valid = [
            (s, r, e)
            for s, r, e in zip(scores, recommendations, explanations)
            if not math.isinf(float(s))
        ]
        if not valid:
            return None
        scores, recommendations, explanations = zip(*valid)

        user_id = request.user_id
        conversation_id = getattr(request, "conversation_id", None)

        result_items = []
        for score, rec_id, raw_exp in zip(scores, recommendations, explanations):
            rec_token = PathLanguageModelingTokenType.ITEM.token + str(rec_id)
            food_item_name = self.decode(rec_token, real_token=True)

            # Build explanation from path tokens
            raw_tokens = raw_exp[1:]  # skip BOS
            real_tokens = [self.decode(t, real_token=True) for t in raw_tokens]
            explanation_text = " → ".join(real_tokens)

            result_items.append({
                "food_item": food_item_name,
                "score": float(score),
                "explanation": explanation_text,
                "food_info": self._lookup_info(food_item_name),
            })

        return {
            "user_id": user_id,
            "recommendations": result_items,
            "conversation_id": conversation_id,
        }

    # -- endpoints -------------------------------------------------------------

    def info(self, request):
        """Look up a food item by name and return its metadata."""
        if isinstance(request, str):
            food_item = request
        else:
            food_item = request.food_item
        return self._lookup_info(food_item)

    def _lookup_info(self, name: str):
        if not name:
            return None
        recipe_id = self._catalog.name_index.get(name.lower())
        if recipe_id is None:
            return None
        record = self._catalog.items.get(recipe_id)
        if record is None:
            return None
        return _item_to_info(record)

    def search(self, request):
        """Search the recipe catalog by name and tags.

        :param request: Search request with ``query`` and ``limit``.
        :returns: Dict with a ``results`` list of ``InfoResponse`` dicts.
        :rtype: dict
        """
        query = request.query
        limit = int(request.limit)

        hits = self._catalog.search(query, limit=limit, tags_field="tags")
        results = []
        for hit in hits:
            record = hit["record"]
            results.append(_item_to_info(record))

        return {"results": results}
