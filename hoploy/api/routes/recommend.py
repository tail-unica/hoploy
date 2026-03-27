from fastapi import APIRouter, Request, Depends, HTTPException, status

from hoploy.core.registry import PluginRegistry
from hoploy import logger

router = APIRouter()

def _get_service(request: Request):
    service = getattr(request.app.state, "service", None)
    if not service or not service.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service warming up")
    return service

@router.post("/recommend")
async def get_recommendation(
    request: Request,
    service = Depends(_get_service)
):
    """
    Place recommendation endpoint.
    """
    body = await request.json()
    RequestSchema = PluginRegistry.get_schema("RecommendationRequest")
    ResponseSchema = PluginRegistry.get_schema("RecommendationResponse")

    req = RequestSchema.model_validate(body)
    logger.info(f"RecommendationRequest from {req.user_id}")

    result = service.run(**req.model_dump())

    if not result:
        raise HTTPException(
            status_code=404,
            detail="No recommendations found with the specified criteria",
        )

    return ResponseSchema.model_validate(result)