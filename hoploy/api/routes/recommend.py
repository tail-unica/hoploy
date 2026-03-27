from fastapi import APIRouter, Request, Depends, HTTPException, status

from hoploy.api.schemas.autism_schema import (
    RecommendationRequest,
    RecommendationResponse,
)

router = APIRouter()

def _get_service(request: Request):
    service = getattr(request.app.state, "service", None)
    if not service or not service.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service warming up")
    return service

@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendation(
    request: RecommendationRequest,
    service = Depends(_get_service)
) -> RecommendationResponse:
    """
    Place recommendation endpoint.
    """

    service.logger.info(f"RecommendationRequest from {request.user_id} with preferences: {request.preferences}")

    recommender_response = service.run(**request.model_dump())

    if not recommender_response:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations found for user {request.user_id} with the specified criteria",
        )

    return RecommendationResponse(**recommender_response)