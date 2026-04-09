import math

from hopwise.utils import PathLanguageModelingTokenType

from hoploy.components import DefaultHopwiseWrapper
from hoploy.registry import Wrapper

from hoploy import logger
from hoploy.core.catalog import get_catalog

from .processors import user_sample_compatible_features


def _sensory_features_for(catalog, poi_id: str) -> list[dict]:
    """Extract sensory features for *poi_id* from catalog neighbors.

    Parses ``SensoryFeature.NOISE.2.3`` tail IDs stored in
    ``catalog.neighbors[poi_id]["HAS_SENSORY_FEATURE"]``.
    """
    tails = catalog.neighbors.get(poi_id, {}).get("HAS_SENSORY_FEATURE", [])
    result = []
    for tail_id in tails:
        parts = tail_id.split(".")
        if len(parts) >= 4:
            feature_name = parts[1].lower()
            rating = float(f"{parts[2]}.{parts[3]}")
            result.append({"feature_name": feature_name, "rating": rating})
    return result


# ---- Explanation helpers ----

sensory_features_it: dict[str, str] = {
    "LIGHT": "luminosità",
    "SPACE": "spazi",
    "CROWD": "affollamento",
    "NOISE": "rumore",
    "ODOR": "odori",
}


def _entity_to_italian(entity_name: str) -> str:
    """Translate a sensory-feature entity name to Italian.

    :param entity_name: Dotted entity name, e.g. ``"SensoryFeature.NOISE"``.
    :type entity_name: str
    :returns: Italian translation or the original name.
    :rtype: str
    """
    parts = entity_name.split(".")
    if len(parts) >= 2 and parts[0] == "SensoryFeature":
        return sensory_features_it.get(parts[1], entity_name)
    return entity_name


def _entity_target_key(entity_name: str) -> str:
    """Extract the feature key from a dotted entity name.

    :param entity_name: Dotted entity name, e.g. ``"SensoryFeature.NOISE"``.
    :type entity_name: str
    :returns: The feature key (e.g. ``"NOISE"``).
    :rtype: str
    """
    parts = entity_name.split(".")
    if len(parts) >= 2 and parts[0] == "SensoryFeature":
        return parts[1]
    return entity_name


def _match_force_path(raw_tokens, force_paths, dataset):
    """Match a sequence's relation tokens against forced path patterns.

    :param raw_tokens: Decoded token strings from a generated sequence.
    :type raw_tokens: list[str]
    :param force_paths: List of relation-name patterns to match.
    :type force_paths: list[list[str]]
    :param dataset: The Hopwise dataset instance.
    :returns: Index of the matched pattern, or ``-1``.
    :rtype: int
    """
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
    """Format a human-readable explanation from a template and tokens.

    :param template_config: Either a template string or a dict with
        ``target`` and ``template`` keys.
    :param raw_tokens: Raw Hopwise token strings.
    :type raw_tokens: list[str]
    :param real_tokens: Human-readable decoded tokens.
    :type real_tokens: list[str]
    :param better_readability: Wrap item names in ``**bold**``.
    :type better_readability: bool
    :returns: The formatted explanation string.
    :rtype: str
    """
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


def _parse_coordinates(coord_str: str):
    """Parse a ``'lon, lat'`` string into a GeoJSON Feature dict.

    :param coord_str: Comma-separated longitude and latitude.
    :type coord_str: str
    :returns: A GeoJSON Feature dict, or ``None`` on parse failure.
    :rtype: dict | None
    """
    if not coord_str or not coord_str.strip():
        return None
    parts = coord_str.split(",")
    if len(parts) != 2:
        return None
    try:
        lon, lat = float(parts[0].strip()), float(parts[1].strip())
    except ValueError:
        return None
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lat, lon]},
        "properties": {},
    }


def _haversine_meters(lat1, lon1, lat2, lon2):
    """Return the great-circle distance in metres between two points.

    :param lat1: Latitude of the first point.
    :type lat1: float
    :param lon1: Longitude of the first point.
    :type lon1: float
    :param lat2: Latitude of the second point.
    :type lat2: float
    :param lon2: Longitude of the second point.
    :type lon2: float
    :returns: Distance in metres.
    :rtype: float
    """
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _item_to_info(record, sensory_features):
    """Convert a catalog record into an ``InfoResponse``-compatible dict.

    :param record: Raw item record from the ``.item`` file.
    :type record: dict
    :param sensory_features: List of sensory feature dicts.
    :type sensory_features: list[dict]
    :returns: A dict matching the ``InfoResponse`` schema.
    :rtype: dict
    """
    name = record.get("name", "").strip()
    address = record.get("address", "").strip() or None
    tags = record.get("tags", "").strip() or None
    coordinates = _parse_coordinates(record.get("coordinates", ""))

    return {
        "place": name,
        "category": tags,
        "address": address,
        "coordinates": coordinates,
        "sensory_features": sensory_features or [],
    }


# ---- Autism model wrapper ----

@Wrapper("autism_model")
class AutismWrapper(DefaultHopwiseWrapper):
    """Autism-specific recommendation wrapper.

    Extends the default wrapper with an item catalog, sensory
    feature awareness, and Italian text explanations.

    :param cfg: Plugin configuration.
    :type cfg: Config
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._catalog = get_catalog(cfg.dataset)
        logger.info(f"Loaded item catalog: {len(self._catalog.items)} items")

    def _names_to_poi_ids(self, names: list) -> list:
        """Convert place names to dataset POI ids.

        Unknown names are skipped with a warning.

        :param names: Human-readable place names.
        :type names: list[str]
        :returns: Matching POI ids.
        :rtype: list
        """
        result = []
        for name in names:
            poi_id = self._catalog.name_index.get(name.lower())
            if poi_id is None:
                logger.warning(f"Place name '{name}' not found in catalog, skipping.")
            else:
                result.append(poi_id)
        return result

    def distill(self, request):
        """Build inputs from preferences and compatible sensory features.

        Converts preferred place names and sensory-compatible features
        into Hopwise-encoded token sequences for zero-shot generation.

        :param request: Incoming recommendation request.
        :type request: Config
        :returns: List of tokenised input strings.
        :rtype: list[str]
        """
        preferences = request.preferences
        aversions = getattr(request, "aversions", None)
        if not preferences:
            logger.error("No preferences provided for zero-shot recommendation.")
            return []

        separator = self.dataset.path_token_separator
        bos = self.dataset.tokenizer.bos_token

        # Convert place names to dataset poi_ids, then to hopwise tokens
        raw_inputs = []
        for poi_id in self._names_to_poi_ids(preferences):
            try:
                token = self.encode(poi_id, PathLanguageModelingTokenType.ITEM.token)
                raw_inputs.append(separator.join([bos, token]))
            except KeyError:
                logger.warning(f"poi_id '{poi_id}' not found in hopwise dataset, skipping.")

        # Compatible sensory feature inputs
        if aversions:
            if not isinstance(aversions, dict):
                aversions = {a["feature_name"]: a["rating"] for a in aversions}
            for feature in user_sample_compatible_features(aversions):
                try:
                    token = self.encode(feature, PathLanguageModelingTokenType.ENTITY.token)
                    raw_inputs.append(separator.join([bos, token]))
                except KeyError:
                    pass

        logger.debug(f"Autism distill: {len(raw_inputs)} raw inputs")
        return raw_inputs

    def handle(self, request):
        """Merge request parameters into runtime configuration.

        :param request: Incoming request.
        :type request: Config
        :returns: Self.
        :rtype: AutismWrapper
        """
        super().handle(request)
        return self

    def expand(self, values, request):
        """Convert raw model output to the autism response schema.

        Builds a ``RecommendationResponse``-compatible dict with place
        names, scores, explanation text, and item metadata.

        :param values: Tuple of ``(scores, recommendations, explanations)``
            or ``None``.
        :param request: The original recommendation request.
        :type request: Config
        :returns: Response dict or ``None`` when no valid results exist.
        :rtype: dict | None
        """
        if values is None:
            return None

        scores, recommendations, explanations = values

        # Filter inf scores
        valid = [(s, r, e) for s, r, e in zip(scores, recommendations, explanations) if not math.isinf(float(s))]
        if not valid:
            return None
        scores, recommendations, explanations = zip(*valid)

        dataset = self.dataset
        force_paths = list(self.cfg.force_paths or [])
        force_path_explanations = list(self.cfg.force_path_explanations or [])
        better_readability = bool(self.cfg.better_readability)

        user_id = request.user_id
        conversation_id = getattr(request, "conversation_id", None)

        result_items = []
        for score, rec_id, raw_exp in zip(scores, recommendations, explanations):
            # Resolve item name
            rec_token = PathLanguageModelingTokenType.ITEM.token + str(rec_id)
            place_name = self.decode(rec_token, real_token=True)

            # Build explanation
            raw_tokens = raw_exp[1:]  # skip BOS
            real_tokens = [self.decode(t, real_token=True) for t in raw_tokens]

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
                "metadata": self.info(place_name),
            })

        return {
            "user_id": user_id,
            "recommendations": result_items,
            "conversation_id": conversation_id,
        }

    def info(self, request):
        """Look up a place by name and return its metadata.

        :param request: A request with an ``item`` attribute, or a
            plain string place name.
        :returns: ``InfoResponse``-compatible dict, or ``None``.
        :rtype: dict | None
        """
        if isinstance(request, str):
            item = request
        else:
            item = request.item
        if not item:
            return None
        poi_id = self._catalog.name_index.get(item.lower())
        if poi_id is None:
            return None
        record = self._catalog.items.get(poi_id)
        if record is None:
            return None
        return _item_to_info(record, _sensory_features_for(self._catalog, poi_id))

    def search(self, request):
        """Search the item catalog by name, tags, position, and categories.

        :param request: Search request with ``query``, ``limit``,
            optional ``position``, ``distance``, and ``categories``.
        :type request: Config
        :returns: Dict with a ``results`` list of ``InfoResponse`` dicts.
        :rtype: dict
        """
        query = request.query
        limit = int(request.limit)
        position = getattr(request, "position", None)
        distance = float(request.distance)
        categories = getattr(request, "categories", None)
        query_lower = query.lower()
        query_terms = [t.strip() for t in query_lower.replace(",", " ").split() if t.strip()]

        # Extract position lat/lon if provided
        ref_lat = ref_lon = None
        if position is not None:
            try:
                geom = position if isinstance(position, dict) else position.model_dump()
                coords = geom["geometry"]["coordinates"]
                ref_lat, ref_lon = float(coords[0]), float(coords[1])
            except (KeyError, IndexError, TypeError, ValueError):
                pass

        results = []
        for poi_id, record in self._catalog.items.items():
            name = record.get("name", "").strip()
            tags = record.get("tags", "").strip()

            # Category filter
            if categories:
                item_tags_lower = tags.lower()
                if not any(cat.lower() in item_tags_lower for cat in categories):
                    continue

            # Text match on name and tags
            searchable = f"{name} {tags}".lower()
            if not any(term in searchable for term in query_terms):
                continue

            # Distance filter
            coord_geo = _parse_coordinates(record.get("coordinates", ""))
            if ref_lat is not None and coord_geo is not None:
                item_lat = coord_geo["geometry"]["coordinates"][0]
                item_lon = coord_geo["geometry"]["coordinates"][1]
                dist = _haversine_meters(ref_lat, ref_lon, item_lat, item_lon)
                if dist > distance:
                    continue

            results.append(_item_to_info(record, _sensory_features_for(self._catalog, poi_id)))

            if len(results) >= limit:
                break

        return {"results": results}