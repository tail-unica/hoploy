import logging
import datetime
from functools import lru_cache
import pathlib

from hopwise.utils import PathLanguageModelingTokenType


_encode_lookup_cache: dict[int, dict[str, dict[str, int]]] = {}


def _get_encode_lookup(dataset, token_type):
    """Return a cached ``{value: index}`` dict for the given token type.

    :param dataset: The Hopwise dataset instance.
    :param token_type: One of the ``PathLanguageModelingTokenType`` token prefixes.
    :type token_type: str
    :returns: Mapping from dataset value to internal index.
    :rtype: dict[str, int]
    """
    ds_id = id(dataset)
    if ds_id not in _encode_lookup_cache:
        _encode_lookup_cache[ds_id] = {}
    cache = _encode_lookup_cache[ds_id]
    if token_type not in cache:
        field_map = {
            PathLanguageModelingTokenType.ITEM.token: dataset.iid_field,
            PathLanguageModelingTokenType.ENTITY.token: dataset.entity_field,
            PathLanguageModelingTokenType.RELATION.token: dataset.relation_field,
            PathLanguageModelingTokenType.USER.token: dataset.uid_field,
        }
        field = field_map[token_type]
        cache[token_type] = {
            tok: idx for idx, tok in enumerate(dataset.field2id_token[field])
        }
    return cache[token_type]


def hopwise_encode(dataset, value, token_type):
    """Encode a dataset ID to a Hopwise token string.

    Example: ``"55"`` → ``"I1"``.

    :param dataset: The Hopwise dataset instance.
    :param value: The raw dataset ID (e.g. a POI id string).
    :param token_type: Token prefix (e.g. ``PathLanguageModelingTokenType.ITEM.token``).
    :type token_type: str
    :returns: The encoded token string.
    :rtype: str
    """
    return token_type + str(_get_encode_lookup(dataset, token_type)[value])


def hopwise_decode(dataset, token, real_token=False):
    """Decode a Hopwise token string back to a dataset ID.

    Example: ``"I1"`` → ``"55"``.

    :param dataset: The Hopwise dataset instance.
    :param token: The encoded token string.
    :type token: str
    :param real_token: If ``True``, resolve item tokens to their
        human-readable name instead of the raw ID.
    :type real_token: bool
    :returns: The decoded dataset value.
    :rtype: str
    """
    if token.startswith(PathLanguageModelingTokenType.ITEM.token) and real_token:
        iid = int(token[1:])
        name_field = dataset.field2id_token["name"][dataset.item_feat[iid]["name"]]
        return " ".join(n for n in name_field if n != "[PAD]")
    if token.startswith(PathLanguageModelingTokenType.ITEM.token):
        return dataset.field2id_token[dataset.iid_field][int(token[1:])]
    elif token.startswith(PathLanguageModelingTokenType.ENTITY.token):
        return dataset.field2id_token[dataset.entity_field][int(token[1:])]
    elif token.startswith(PathLanguageModelingTokenType.RELATION.token):
        return dataset.field2id_token[dataset.relation_field][int(token[1:])]
    elif token.startswith(PathLanguageModelingTokenType.USER.token):
        return dataset.field2id_token[dataset.uid_field][int(token[1:])]
    return token


def id2tokenizer_token(dataset, ids, token_type):
    """Convert a list of dataset IDs to tokenizer token IDs.

    :param dataset: The Hopwise dataset instance.
    :param ids: Dataset-level identifiers to convert.
    :type ids: list
    :param token_type: One of ``'item'``, ``'entity'``, ``'relation'``,
        ``'user'``.
    :type token_type: str
    :returns: List of integer token IDs recognised by the tokenizer.
    :rtype: list[int]
    """
    type_map = {
        "item": PathLanguageModelingTokenType.ITEM.token,
        "entity": PathLanguageModelingTokenType.ENTITY.token,
        "relation": PathLanguageModelingTokenType.RELATION.token,
        "user": PathLanguageModelingTokenType.USER.token,
    }
    prefix = type_map.get(token_type, token_type)
    result = []
    for v in ids:
        try:
            hopwise_token = hopwise_encode(dataset, v, prefix)
            tid = dataset.tokenizer.convert_tokens_to_ids(hopwise_token)
            result.append(tid)
        except (KeyError, IndexError):
            pass
    return result


class CustomFormatter(logging.Formatter):
    """ANSI-coloured log formatter.

    Applies terminal colour codes based on the log level.
    """
    # Use standard ANSI color codes to ensure terminal support
    grey = "\x1b[90m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt: str, datefmt: str | None = None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._level_formats = {
            logging.DEBUG: f"{self.grey}{fmt}{self.reset}",
            logging.INFO: fmt,
            logging.WARNING: f"{self.yellow}{fmt}{self.reset}",
            logging.ERROR: f"{self.red}{fmt}{self.reset}",
            logging.CRITICAL: f"{self.bold_red}{fmt}{self.reset}",
        }

    def format(self, record):
        record.levelname_c = f"{record.levelname}:"
        log_fmt = self._level_formats.get(record.levelno, self._fmt)
        formatter = logging.Formatter(log_fmt, self.datefmt)
        return formatter.format(record)

@lru_cache(maxsize=1)
def get_logger(cfg) -> logging.Logger:
    """Create (or return the cached) root ``hoploy`` logger.

    :param cfg: Logging configuration section with ``level`` and ``format``.
    :type cfg: ~hoploy.core.config.Config
    :returns: A configured :class:`logging.Logger`.
    :rtype: logging.Logger
    """
    logfile = pathlib.Path("logs") / f"core-{datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("hoploy")
    logger.setLevel(cfg.level)

    if not logger.handlers:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(CustomFormatter(cfg.format))

        sh = logging.StreamHandler()
        sh.setLevel(cfg.level)
        sh.setFormatter(CustomFormatter(cfg.format))

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger