"""Item catalog with Parquet-backed persistence.

Provides a :class:`Catalog` that loads ``.item``, ``.kg`` and link
files from a Hopwise dataset directory.  On first access the raw TSV
files are parsed and persisted as a ``.catalog.parquet`` file so that
subsequent startups skip the expensive CSV parsing entirely.
"""

import csv
import difflib
import json
import logging
import os
import pathlib
from collections import defaultdict
from functools import lru_cache

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Directory where catalog parquet files are cached.
# Override via HOPLOY_CATALOG_DIR env var; defaults to /app/catalog.
_DEFAULT_CATALOG_DIR = "/app/catalog"


def _catalog_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("HOPLOY_CATALOG_DIR", _DEFAULT_CATALOG_DIR))


def _find_kg_stem(ds_dir: pathlib.Path) -> str:
    """Return the stem of the first ``.kg`` file found in *ds_dir*.

    For example, ``/app/dataset/autism.kg`` → ``"autism"``.
    Falls back to the directory name if no ``.kg`` file exists.
    """
    for kg_file in ds_dir.glob("*.kg"):
        return kg_file.stem
    return ds_dir.name


class Catalog:
    """Read-only view of a Hopwise dataset's extended item data.

    :param items: Mapping of item id to full record dict.
    :param name_index: Mapping of lowercase item name to item id.
    :param neighbors: Mapping of item id to ``{relation: [tail_id, …]}``.
    """

    __slots__ = ("_items", "_name_index", "_neighbors", "_valid_names")

    def __init__(self, items: dict, name_index: dict, neighbors: dict):
        self._items = items
        self._name_index = name_index
        self._neighbors = neighbors
        self._valid_names: list[str] | None = None

    @property
    def items(self) -> dict:
        return self._items

    @property
    def name_index(self) -> dict:
        return self._name_index

    @property
    def neighbors(self) -> dict:
        return self._neighbors

    def resolve_to_valid(self, name: str, valid_ids: set[str], top_k: int = 1, cutoff: float = 0.6) -> list[str]:
        """Resolve *name* to item ids present in *valid_ids*.

        1. Exact lookup via :attr:`name_index` — if the id is in
           *valid_ids*, return immediately.
        2. Otherwise, fuzzy-match *name* against all catalog names whose
           ids belong to *valid_ids* using :func:`difflib.get_close_matches`.

        :param name: Human-readable item name.
        :param valid_ids: Set of item ids known to the model vocabulary.
        :param top_k: Maximum number of results to return.
        :param cutoff: Similarity threshold for fuzzy matching (0–1).
        :returns: Up to *top_k* item ids, best match first.  Empty list
            if nothing matches above the cutoff.
        """
        # 1. Exact match
        item_id = self._name_index.get(name.lower())
        if item_id is not None and item_id in valid_ids:
            return [item_id]

        # 2. Build valid-name list lazily (cached per valid_ids set)
        valid_name_map = {
            n: iid for n, iid in self._name_index.items() if iid in valid_ids
        }
        if not valid_name_map:
            return []

        matches = difflib.get_close_matches(
            name.lower(), valid_name_map.keys(), n=top_k, cutoff=cutoff,
        )
        return [valid_name_map[m] for m in matches]

    def search(self, query: str, limit: int = 10, tags_field: str = "tags") -> list[dict]:
        """Full-text search over item names and tags.

        :param query: Space/comma-separated search terms.
        :param limit: Maximum results to return.
        :param tags_field: Name of the tags column in item records.
        :returns: List of ``{item_id, record}`` dicts.
        """
        terms = [t.strip() for t in query.lower().replace(",", " ").split() if t.strip()]
        if not terms:
            return []

        results = []
        for item_id, record in self._items.items():
            name = record.get("name", "").strip()
            tags = record.get(tags_field, "").strip()
            searchable = f"{name} {tags}".lower()
            if any(term in searchable for term in terms):
                results.append({"item_id": item_id, "record": record})
                if len(results) >= limit:
                    break
        return results


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _find_link_file(ds_dir: pathlib.Path, ds_name: str) -> pathlib.Path | None:
    for candidate in (
        ds_dir / f"{ds_name}.item_link",
        ds_dir / f"{ds_name}.link",
    ):
        if candidate.exists():
            return candidate
    return None


def _parse_tsv(ds_dir: pathlib.Path, ds_name: str):
    """Parse .item, link and .kg files into catalog dicts."""

    # -- .item --
    item_file = ds_dir / f"{ds_name}.item"
    items: dict[str, dict] = {}
    name_index: dict[str, str] = {}

    with open(item_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        col_names = [h.split(":")[0] for h in header]
        id_col = col_names[0]
        for row in reader:
            record = dict(zip(col_names, row))
            item_id = record[id_col]
            items[item_id] = record
            name = record.get("name", "").strip()
            if name:
                name_index[name.lower()] = item_id

    # -- link file (item_id → entity_id) --
    entity_to_item: dict[str, str] = {}
    link_file = _find_link_file(ds_dir, ds_name)
    if link_file is not None:
        with open(link_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            for row in reader:
                item_id, entity_id = row[0], row[1]
                entity_to_item[entity_id] = item_id

    # -- .kg --
    neighbors: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    kg_file = ds_dir / f"{ds_name}.kg"
    if kg_file.exists():
        with open(kg_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            for row in reader:
                if len(row) < 3:
                    continue
                head_id, relation, tail_id = row[0], row[1], row[2]
                resolved_id = entity_to_item.get(head_id, head_id)
                neighbors[resolved_id][relation].append(tail_id)

    # Freeze inner defaultdicts
    neighbors = {k: dict(v) for k, v in neighbors.items()}

    return items, name_index, neighbors


def _to_parquet(items: dict, name_index: dict, neighbors: dict, path: pathlib.Path):
    """Persist catalog data as a single Parquet file.

    Each item becomes one row.  The ``neighbors`` dict is stored as a
    JSON string column so the schema stays flat and independent of the
    set of relations present in the KG.
    """
    ids = list(items.keys())
    records_json = [json.dumps(items[i], ensure_ascii=False) for i in ids]
    neighbors_json = [json.dumps(neighbors.get(i, {}), ensure_ascii=False) for i in ids]
    names = [items[i].get("name", "").strip() for i in ids]

    table = pa.table({
        "item_id": ids,
        "name": names,
        "record": records_json,
        "neighbors": neighbors_json,
    })
    pq.write_table(table, str(path), compression="snappy")
    logger.info("Wrote catalog parquet: %s (%d items)", path, len(ids))


def _from_parquet(path: pathlib.Path):
    """Reconstruct catalog dicts from a Parquet file."""
    table = pq.read_table(str(path))
    item_ids = table.column("item_id").to_pylist()
    names = table.column("name").to_pylist()
    records_json = table.column("record").to_pylist()
    neighbors_json = table.column("neighbors").to_pylist()

    items: dict[str, dict] = {}
    name_index: dict[str, str] = {}
    neighbors: dict[str, dict[str, list[str]]] = {}

    for item_id, name, rec_j, neigh_j in zip(item_ids, names, records_json, neighbors_json):
        items[item_id] = json.loads(rec_j)
        if name:
            name_index[name.lower()] = item_id
        neigh = json.loads(neigh_j)
        if neigh:
            neighbors[item_id] = neigh

    return items, name_index, neighbors


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_catalog(dataset_path: str = "dataset") -> Catalog:
    """Return a :class:`Catalog` for *dataset_path*.

    On first call for a given path the function checks the catalog volume
    (``HOPLOY_CATALOG_DIR``, default ``/app/catalog``) for a cached
    ``<dataset>.parquet`` file named after the ``.kg`` file found in
    *dataset_path*.  If absent it parses the raw TSV files and writes the
    Parquet to the catalog volume for future startups.  The dataset directory
    is never written to, so it can be mounted read-only.  Results are cached
    in-process via :func:`functools.lru_cache`.

    :param dataset_path: Path to the dataset directory.
    """
    ds_dir = pathlib.Path(dataset_path)
    ds_name = _find_kg_stem(ds_dir)

    cat_dir = _catalog_dir()
    cat_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = cat_dir / f"{ds_name}.parquet"

    if parquet_path.exists():
        logger.info("Loading catalog from cache: %s", parquet_path)
        items, name_index, neighbors = _from_parquet(parquet_path)
    else:
        logger.info(
            "Parsing TSV catalog for '%s' (will cache to %s)", ds_name, parquet_path
        )
        items, name_index, neighbors = _parse_tsv(ds_dir, ds_name)
        _to_parquet(items, name_index, neighbors, parquet_path)

    return Catalog(items, name_index, neighbors)
