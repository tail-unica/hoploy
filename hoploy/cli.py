"""hoploy CLI — thin wrapper around docker compose.

Commands
--------
serve   Start one or more plugin services (docker compose up).
train   Run the training service (docker compose run --rm train).

Examples
--------
  hoploy serve hummus
  hoploy serve hummus autism
  hoploy serve hummus --detach --build
  hoploy train -c autism_test_hoploy -d autism
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _compose(*args: str) -> int:
    """Run ``docker compose <args>`` in the current working directory."""
    result = subprocess.run(["docker", "compose", *args])
    return result.returncode


def _serve(args: argparse.Namespace) -> int:
    """Start one or more plugin services via ``docker compose up``."""
    flags: list[str] = []
    if args.detach:
        flags.append("--detach")
    if args.build:
        flags.append("--build")
    return _compose("up", *flags, *args.plugins)


def _train(args: argparse.Namespace) -> int:
    """Run the training container via ``docker compose run --rm train``."""
    return _compose(
        "run", "--rm", "train",
        "-c", args.checkpoint,
        "--dataset", args.dataset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hoploy",
        description="hoploy — docker compose wrapper for plugin serving and training",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # ── serve ────────────────────────────────────────────────────────────────
    serve_p = sub.add_parser(
        "serve",
        help="Start one or more plugin services",
        description=(
            "Start the named plugin services via docker compose up.\n\n"
            "Each name must match a service defined in compose.yaml.\n\n"
            "Examples:\n"
            "  hoploy serve hummus\n"
            "  hoploy serve hummus autism\n"
            "  hoploy serve hummus --detach"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    serve_p.add_argument(
        "plugins",
        nargs="+",
        metavar="plugin",
        help="Plugin service name(s) defined in compose.yaml (e.g. hummus autism)",
    )
    serve_p.add_argument(
        "--detach", "-D",
        action="store_true",
        help="Run containers in the background (passes --detach to docker compose up)",
    )
    serve_p.add_argument(
        "--build",
        action="store_true",
        help="Force a rebuild of the image before starting",
    )
    serve_p.set_defaults(func=_serve)

    # ── train ────────────────────────────────────────────────────────────────
    train_p = sub.add_parser(
        "train",
        help="Run the Hopwise training service",
        description=(
            "Run the training container via docker compose run --rm train.\n\n"
            "Examples:\n"
            "  hoploy train -c autism_test_hoploy -d autism\n"
            "  hoploy train -c hummus_pearlm -d hummus"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_p.add_argument(
        "-c", "--checkpoint",
        required=True,
        metavar="CHECKPOINT_CONFIG",
        help="Checkpoint config name (subdirectory under checkpoints/)",
    )
    train_p.add_argument(
        "-d", "--dataset",
        required=True,
        metavar="DATASET",
        help="Dataset name (subdirectory under datasets/)",
    )
    train_p.set_defaults(func=_train)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
