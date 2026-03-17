from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hoploy.api.schemas.request import RecommendRequest
from hoploy.api.schemas.response import RecommendResponse
from hoploy.core.state import GenerationState

router = APIRouter(tags=["recommend"])


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(body: RecommendRequest, request: Request) -> Any:
    """Generate recommendations for a user."""
    pipeline = request.app.state.pipeline

    if pipeline is None or not getattr(pipeline, "_built", False):
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    state = GenerationState(
        user_id=body.user_id,
        step=0,
        input_ids=(),
        generated_ids=(),
        constraint_config={},
        context=body.model_dump(),
    )

    result = await asyncio.to_thread(pipeline.run, state)

    if result is None:
        raise HTTPException(status_code=500, detail="Generation failed")

    return result
