"""
src/api/routes/agent.py
========================
FastAPI route for the AirSentinel ReAct AI Analyst agent.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.react_agent import AirSentinelAgent

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


# ── 1. Singleton Dependency ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_agent() -> AirSentinelAgent:
    """Instantiate and cache the AirSentinelAgent instance as a singleton."""
    logger.info("Initializing singleton AirSentinelAgent (provider='groq')...")
    return AirSentinelAgent(provider="groq")


# ── 2. Schemas ────────────────────────────────────────────────────────────────

class AgentQuery(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language query for the air quality agent.",
        json_schema_extra={
            "example": "Which monitoring zone currently exhibits the highest PM2.5 levels?"
        },
    )


class AgentResponse(BaseModel):
    response: str
    status: str = "success"


# ── 3. Route Handler ──────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
    summary="Query AI Analyst Agent",
)
async def query_agent(
    body: AgentQuery,
    agent: Annotated[AirSentinelAgent, Depends(get_agent)],
):
    """Query the AirSentinel ReAct Agent for real-time sensor insights and forecasts."""
    cleaned_query = body.query.strip()
    if not cleaned_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty or whitespace.",
        )

    logger.info(f"Received agent query: '{cleaned_query}'")

    try:
        # Direct, reliable offloading of the synchronous AirSentinelAgent.run method
        result = await run_in_threadpool(agent.run, cleaned_query)

        logger.info("Agent successfully generated response.")
        return AgentResponse(response=str(result), status="success")

    except Exception as e:
        logger.error(f"Agent execution error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}",
        )