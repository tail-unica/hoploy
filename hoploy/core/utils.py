import logging
import datetime
from functools import lru_cache
import pathlib

from hopwise.utils import PathLanguageModelingTokenType


def hopwise_encode(dataset, value, token_type):
    """Encode a dataset ID to a hopwise token string (e.g. "55" -> "I1")."""
    lookups = {
        PathLanguageModelingTokenType.ITEM.token: lambda: {
            tok: idx for idx, tok in enumerate(dataset.field2id_token[dataset.iid_field])
        },
        PathLanguageModelingTokenType.ENTITY.token: lambda: {
            tok: idx for idx, tok in enumerate(dataset.field2id_token[dataset.entity_field])
        },
        PathLanguageModelingTokenType.RELATION.token: lambda: {
            tok: idx for idx, tok in enumerate(dataset.field2id_token[dataset.relation_field])
        },
        PathLanguageModelingTokenType.USER.token: lambda: {
            tok: idx for idx, tok in enumerate(dataset.field2id_token[dataset.uid_field])
        },
    }
    return token_type + str(lookups[token_type]()[value])


def hopwise_decode(dataset, token):
    """Decode a hopwise token string back to a dataset ID (e.g. "I1" -> "55")."""
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
    
    :param token_type: One of 'item', 'entity', 'relation', 'user'
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
    logfile = pathlib.Path("logs") / f"core-{datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("AutismRecsysAPI")
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