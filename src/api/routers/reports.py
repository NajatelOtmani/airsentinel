from fastapi import APIRouter, Depends

from src.agents.report_generator import AutoReportGenerator
from src.api.auth import get_current_user

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])
_gen = AutoReportGenerator(provider="groq")


@router.post("/{location_id}")
async def generate_report(location_id: str, user: str = Depends(get_current_user)):
    # generate_report() itself now catches agent failures internally and
    # returns a structured dict either way (see report_generator.py) —
    # this route just needs to relay that, rather than assuming success
    # and discarding the actual content like before.
    report_data = _gen.generate_report(location_id=location_id)
    return report_data
