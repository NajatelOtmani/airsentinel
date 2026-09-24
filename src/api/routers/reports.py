from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from src.agents.report_generator import AutoReportGenerator
from src.api.auth import get_current_user

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])
_gen = AutoReportGenerator(provider="groq")


@router.post("/{location_id}")
async def generate_report(location_id: str, user: str = Depends(get_current_user)):
    # generate_report() catches agent failures internally and returns a
    # structured dict either way (see report_generator.py) — this route
    # just needs to relay that.
    #
    # generate_report() is synchronous and blocking (pandas CSV reads +
    # network calls to the LLM provider), so it's run in a threadpool
    # rather than awaited directly. Calling a blocking function straight
    # inside `async def` would stall FastAPI's single event loop for the
    # duration of every report — freezing all other in-flight requests —
    # not raise an error, so it's easy to miss in testing but shows up
    # immediately under real concurrent load.
    report_data = await run_in_threadpool(_gen.generate_report, location_id=location_id)
    return report_data