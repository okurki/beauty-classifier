from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.infrastructure.database import DB
from src.infrastructure.ml_models import load_models
from src.interfaces.api.v1.routers import routers
from src.config import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    async with DB.lifespan():
        yield


def create_app(lifespan: AsyncGenerator = lifespan) -> FastAPI:
    app = FastAPI(
        lifespan=lifespan,
        docs_url=None if config.prod else "/docs",
        redoc_url=None if config.prod else "/redoc",
        openapi_url=None if config.prod else "/openapi.json",
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

    for router in routers:
        app.include_router(router)

    app.mount("/", StaticFiles(directory="static", html=True), name="welcome page")
    app.mount(
        "/celebrities",
        StaticFiles(directory="datasets/open_famous_people_faces"),
        name="celebrities",
    )

    return app


if __name__ == "__main__":
    uvicorn.run(
        create_app(lifespan),
        host=config.api.host,
        port=config.api.port,
        log_config=config.logging.config,
    )
