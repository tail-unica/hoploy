from typing import List, Optional

from pydantic import BaseModel, Field, PositiveInt

from hoploy import Request, Response


# --- Shared Models ---


class IngredientList(BaseModel):
    ingredients: list[str] = Field(
        description="List of ingredients",
    )
    quantities: Optional[list[str]] = Field(
        default=None,
        description="Optional list of quantities for each ingredient",
    )


class HealthinessInfo(BaseModel):
    score: str = Field(
        description="Nutri-Score category (A–E)",
    )


class SustainabilityInfo(BaseModel):
    score: str = Field(
        description="Sustainability score category (A–E)",
    )
    CF: Optional[float] = Field(
        default=None,
        description="Carbon Footprint estimate",
    )
    WF: Optional[float] = Field(
        default=None,
        description="Water Footprint estimate",
    )


# --- INFO ---


@Request("info")
class InfoRequest(BaseModel):
    food_item: str = Field(
        description="Name of the food item to get information about",
        example="Spaghetti Carbonara",
    )


@Response("info")
class InfoResponse(BaseModel):
    food_item: str = Field(description="Name of the food item")
    food_item_type: str = Field(description="Type of food item (recipe or ingredient)")
    healthiness: Optional[HealthinessInfo] = None
    sustainability: Optional[SustainabilityInfo] = None
    nutritional_values: Optional[dict[str, Optional[float]]] = None
    nutritional_value_groups: Optional[dict[str, Optional[str]]] = None
    ingredients: Optional[IngredientList] = None
    food_item_url: Optional[str] = None


# --- SEARCH ---


@Request("search")
class SearchRequest(BaseModel):
    """Request model for the food item search endpoint."""

    query: str = Field(
        description="Search query string",
        example="Cranberry",
    )
    limit: Optional[PositiveInt] = Field(
        default=10,
        description="Maximum number of results to return",
        example=5,
    )


@Response("search")
class SearchResponse(BaseModel):
    """Response model for the food item search endpoint."""

    results: List[InfoResponse] = Field(
        description="List of food items matching the search criteria"
    )


# --- RECOMMENDATION ---


@Request("recommend")
class RecommendationRequest(BaseModel):
    user_id: str = Field(
        description="Unique identifier for the user",
        example="12345",
    )
    preferences: list[str] = Field(
        description="List of food items the user likes",
        example=["Spaghetti Carbonara", "Cranberry-Orange Caramel Corn"],
    )
    previous_recommendations: Optional[list[str]] = Field(
        default=None,
        example=["Almond Butter Cookies"],
        description="List of previously recommended items to avoid repetition",
    )
    hard_restrictions: Optional[list[str]] = Field(
        default=None,
        example=["Peanuts", "Shellfish"],
        description="List of food items to completely exclude from recommendations",
    )
    soft_restrictions: Optional[list[str]] = Field(
        default=None,
        example=["Sugar", "Salt"],
        description="List of food items to penalise in recommendations",
    )
    recommendation_count: Optional[PositiveInt] = Field(
        default=5,
        description="Number of recommendations to return",
    )
    diversity_factor: Optional[float] = Field(
        default=0.5,
        description="Controls how diverse the recommendations should be (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Identifier for the conversation",
    )


class RecommendationItem(BaseModel):
    food_item: str = Field(description="Name of the recommended food item")
    score: float = Field(description="Recommendation score")
    explanation: str = Field(description="Human-readable explanation")
    food_info: Optional[InfoResponse] = None


@Response("recommend")
class RecommendationResponse(BaseModel):
    user_id: str = Field(description="Unique identifier for the user")
    recommendations: list[RecommendationItem] = Field(
        description="List of recommended food items with scores and metadata"
    )
    conversation_id: Optional[str] = None
