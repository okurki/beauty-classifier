from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse

from prometheus_client import REGISTRY
from prometheus_client.openmetrics.exposition import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)

default_router = APIRouter(tags=["Default"])


@default_router.get("/")
def index(request: Request):
    return HTMLResponse(request.app.state.welcome_page_html)


@default_router.get("/health")
async def health():
    return {"status": "ok"}


@default_router.get("/metrics")
def metrics(request: Request):
    return Response(
        generate_latest(REGISTRY), headers={"Content-Type": CONTENT_TYPE_LATEST}
    )
