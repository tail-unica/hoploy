from typing import Literal, Optional, List, Dict, Any
import json
import math
from typing import Any
from pprint import pformat

from fastapi import APIRouter, Request, Depends, HTTPException, status, Query

from hoploy.api.schemas.autism_schema import InfoResponse, SearchRequest, SearchResponse

router = APIRouter()

def _get_service(request: Request):
    service = getattr(request.app.state, "service", None)
    if not service or not service.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service warming up")
    return service

@router.get("/info/{item}", response_model=InfoResponse)
async def get_info(
    item: str,
    service = Depends(_get_service)
) -> InfoResponse:
    """
    Item information endpoint.

    Multiple items can have the same name, so this endpoint returns the first match.
    The same name could refer to different types of items, and without additional context about
    the queried item, the endpoint cannot distinguish between them.

    **item**: Name of the item to get information about
    **model**: Model to use for fetching information
    """
    
    info_response = await service.info(item=item)

    if not info_response:
        raise HTTPException(
            status_code=404,
            detail=f"No information found for item '{item}' with the specified model",
        )

    return InfoResponse(**info_response)


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    service = Depends(_get_service)
) -> SearchResponse:
    """
    Item search endpoint.

    **query**: Search query string
    **limit**: Maximum number of results to return
    **position**: Optional user position for proximity filtering
    **distance**: Optional maximum distance (in meters) from the user position
    **categories**: Optional list of category IDs to filter results
    """

    search_results = await service.search(**request.model_dump())

    def _sanitize_for_json(obj: Any):
        if isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_for_json(v) for v in obj]
        if isinstance(obj, float):
            if not math.isfinite(obj):
                return None
            return round(obj, 1)
        return obj

    raw = search_results if not isinstance(search_results, str) else json.loads(search_results)
    try:
        pretty = json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False)
    except (ValueError, TypeError):
        pretty = json.dumps(_sanitize_for_json(raw), indent=2, sort_keys=True, ensure_ascii=False)

    return SearchResponse(**search_results)