from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AversionDetail(BaseModel):
    feature_name: str
    rating: float


class RecommendRequest(BaseModel):
    """POST /recommend request body."""

    user_id: str
    preferences: list[str] | None = None
    previous_recommendations: list[str] = Field(default_factory=list)
    recommendation_count: int = 5
    diversity_factor: float = 0.5
    aversions: list[AversionDetail] | None = None
    mode: str = Field(
        default="zero_shot",
        description="Generation mode: 'zero_shot' or 'existing_user'.",
    )
