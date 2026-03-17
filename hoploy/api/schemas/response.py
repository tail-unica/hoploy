from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """A single recommendation entry."""

    item_id: str
    score: float
    explanation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendResponse(BaseModel):
    """POST /recommend response body."""

    user_id: str
    recommendations: list[RecommendationItem]
    conversation_id: str | None = None
