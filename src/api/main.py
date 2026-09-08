import asyncio

from dotenv import load_dotenv

load_dotenv()
import asyncio
# ...rest of your existing imports
from contextlib import asynccontextmanager

from fastapi import (Depends, FastAPI, HTTPException, Request, WebSocket,
                     WebSocketDisconnect, status)
from fastapi.security import OAuth2PasswordRequestForm
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.auth import create_access_token, verify_credentials
from src.api.rate_limit import limiter
from src.api.routers import agent, anomalies, forecasts, reports, sensors
from src.api.websocket import manager, poll_anomalies_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    poll_task = asyncio.create_task(poll_anomalies_loop())
    yield
    poll_task.cancel()


app = FastAPI(title="AirSentinel API", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(sensors.router)
app.include_router(anomalies.router)
app.include_router(forecasts.router)
app.include_router(agent.router)
app.include_router(reports.router)


@app.post("/auth/login")
@limiter.limit("60/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    if not verify_credentials(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    token = create_access_token(form_data.username)
    return {"access_token": token, "token_type": "bearer"}


@app.websocket("/ws/alerts")
async def websocket_alerts(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.get("/health")
async def health():
    return {"status": "ok"}
