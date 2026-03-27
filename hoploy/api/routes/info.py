from fastapi import APIRouter, Request, Depends, HTTPException, status

from hoploy.core.registry import PluginRegistry

router = APIRouter()

def _get_service(request: Request):
    service = getattr(request.app.state, "service", None)
    if not service or not service.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service warming up")
    return service

@router.get("/info/{item}")
async def get_info(
    item: str,
    service = Depends(_get_service)
):
    """
    Item information endpoint.
    """
    info_response = await service.info(item=item)

    if not info_response:
        raise HTTPException(
            status_code=404,
            detail=f"No information found for item '{item}'",
        )

    ResponseSchema = PluginRegistry.get_schema("InfoResponse")
    return ResponseSchema.model_validate(info_response)


@router.post("/search")
async def search(
    request: Request,
    service = Depends(_get_service)
):
    """
    Item search endpoint.
    """
    body = await request.json()
    RequestSchema = PluginRegistry.get_schema("SearchRequest")
    ResponseSchema = PluginRegistry.get_schema("SearchResponse")

    req = RequestSchema.model_validate(body)
    search_results = await service.search(**req.model_dump())

    return ResponseSchema.model_validate(search_results)