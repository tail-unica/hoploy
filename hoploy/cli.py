from __future__ import annotations

import argparse
import sys

import uvicorn

from hoploy.config.loader import load_config
from hoploy.api.app import create_app
from hoploy.core.pipeline import Pipeline


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

    print(f"Starting Hoploy with configuration from")
    print(config.to_dict())

    from hoploy.core.registry import _storage, load_plugin

    pipeline = Pipeline(config)

    query = {
        "user_id": "user_123",
        "input": ["48", "21"],
        "previous_recommendations": ["17"],
        "recommendation_count": 5,
        "diversity_factor": 0.7,
    }

    print(pipeline.run(**query))

    # app = create_app(pipeline)

    # uvicorn.run(
    #     app,
    #     host=config.serve.host,
    #     port=config.serve.port,
    #     workers=config.serve.workers,
    #     log_level=config.serve.log_level,
    # )


if __name__ == "__main__":
    sys.exit(main())
