"""Item catalog with Parquet-backed persistence.

Provides a :class:`Catalog` that loads ``.item``, ``.kg`` and link
files from a Hopwise dataset directory.  On first access the raw TSV
files are parsed and persisted as a ``.catalog.parquet`` file so that
subsequent startups skip the expensive CSV parsing entirely.
"""

import csv
import json
import logging
import pathlib
from collections import defaultdict
from functools import lru_cache

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_PARQUET_FILENAME = ".catalog.parquet"


class Catalog:
    """Read-only view of a Hopwise dataset's extended item data.

    :param items: Mapping of item id to full record dict.
    :param name_index: Mapping of lowercase item name to item id.
    :param neighbors: Mapping of item id to ``{relation: [tail_id, …]}``.
    """

    __slots__ = ("_items", "_name_index", "_neighbors")

    def __init__(self, items: dict, name_index: dict, neighbors: dict):
        self._items = items
        self._name_index = name_index
        self._neighbors = neighbors

    @property
    def items(self) -> dict:
        return self._items

    @property
    def name_index(self) -> dict:
        return self._name_index

    @property
    def neighbors(self) -> dict:
        return self._neighbors


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
def get_catalog(dataset_path: str) -> Catalog:
    """Return a :class:`Catalog` for *dataset_path*.

    On first call for a given path the function checks for a cached
    ``.catalog.parquet`` file.  If absent it parses the raw TSV files
    and writes the Parquet for future startups.  Results are cached
    in-process via :func:`functools.lru_cache`.

    :param dataset_path: Path to the dataset directory.
    """
    ds_dir = pathlib.Path(dataset_path)
    ds_name = ds_dir.name
    parquet_path = ds_dir / _PARQUET_FILENAME

    if parquet_path.exists():
        logger.info("Loading catalog from parquet: %s", parquet_path)
        items, name_index, neighbors = _from_parquet(parquet_path)
    else:
        logger.info("Parsing TSV catalog for '%s' (first time, will cache as parquet)", ds_name)
        items, name_index, neighbors = _parse_tsv(ds_dir, ds_name)
        _to_parquet(items, name_index, neighbors, parquet_path)

    return Catalog(items, name_index, neighbors)
