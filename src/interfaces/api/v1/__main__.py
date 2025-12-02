from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging

import uvicorn
import uvicorn.config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.infrastructure.database import DB
from src.infrastructure.ml_models import load_models
from src.interfaces.api.v1.schemas import InferenceRead
from src.application.services.ml import MLService
from src.infrastructure.repositories import InferenceRepository
from src.interfaces.api.v1.routers import routers
from src.interfaces.api.v1.utils import setting_otlp
from src.interfaces.api.v1.middleware import PrometheusMiddleware
from src.config import config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    with open("static/index.html", "r", encoding="utf-8") as f:
        app.state.welcome_page_html = f.read()
    async with DB.lifespan():
        try:
            async with DB.async_sessionmaker() as session:
                ml_service = MLService(InferenceRepository(session), InferenceRead)
                await ml_service.load_feedback_weights()
        except Exception as e:
            logger.warning(f"Could not load feedback weights on startup: {e}")
        yield


def create_app(lifespan: AsyncGenerator = lifespan) -> FastAPI:
    app = FastAPI(
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_origin_regex=None,
        expose_headers=["*"],
    )

    app.add_middleware(PrometheusMiddleware)

    for router in routers:
        app.include_router(router)

    app.mount(
        "/celebrities_pretty",
        StaticFiles(directory="datasets/celebrities_pretty"),
        name="celebrities_pretty",
    )

    return app


if __name__ == "__main__":
    app = create_app(lifespan)
    setting_otlp(app, config.app_name, config.otlp_grpc_endpoint)

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = (
        "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s resource.service.name=%(otelServiceName)s] - %(message)s"
    )

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_config=log_config,
    )
