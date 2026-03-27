from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI

from hoploy.api.routes import health, recommend, info
from hoploy.core.pipeline import Pipeline
from hoploy.config.loader import load_config

@asynccontextmanager
async def lifespan(app: FastAPI):
    service = Pipeline(load_config("configs/default.yaml"))
    app.state.service = service
    try:
        yield
    finally:
        await service.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="Hopwise-Serve", version="0.1.0", lifespan=lifespan)

    # Routes
    app.include_router(health.router)
    app.include_router(info.router)
    app.include_router(recommend.router)

    return app
