from __future__ import annotations

import argparse
import sys

import uvicorn

from hoploy.config.loader import load_config
from hoploy.api.factory import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hoploy",
        description="Hoploy — serving layer for Hopwise recommendation models",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        metavar="FILE",
        help="Path to the YAML config file (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    app = create_app(config)

    uvicorn.run(
        app,
        host=config.serve.host,
        port=config.serve.port,
        workers=config.serve.workers,
        log_level=config.serve.log_level,
    )


if __name__ == "__main__":
    sys.exit(main())
