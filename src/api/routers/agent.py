from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.agents.react_agent import AirSentinelAgent
from src.api.auth import get_current_user

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])
_agent = AirSentinelAgent(provider="groq")


class AgentQuery(BaseModel):
    query: str


@router.post("/")
async def query_agent(body: AgentQuery, user: str = Depends(get_current_user)):
    result = _agent.run(body.query)
    return {"response": result}
