from fastapi import APIRouter, Depends

from src.api.auth import get_current_user

router = APIRouter(prefix="/api/v1/forecasts", tags=["forecasts"])


@router.get("/{location_id}")
async def get_forecast(location_id: str, user: str = Depends(get_current_user)):
    return {
        "location_id": location_id,
        "forecast": "TODO: wire ONNX transformer inference",
    }
