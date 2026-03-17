from __future__ import annotations

from fastapi import APIRouter

from hoploy.core.registry import (
    _MODELS,
    _POSTPROCESSORS,
    _PREPROCESSORS,
    _PROCESSORS,
    _UNPACKERS,
)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/registry")
def list_registry() -> dict:
    """Return names of all registered components."""
    return {
        "models": sorted(_MODELS),
        "preprocessors": sorted(_PREPROCESSORS),
        "processors": sorted(_PROCESSORS),
        "postprocessors": sorted(_POSTPROCESSORS),
        "unpackers": sorted(_UNPACKERS),
    }
