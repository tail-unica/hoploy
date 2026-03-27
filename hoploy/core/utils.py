import logging
import datetime
from functools import lru_cache
import pathlib


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