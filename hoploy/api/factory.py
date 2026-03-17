from __future__ import annotations

from fastapi import FastAPI

from hoploy.api.middleware.logging import RequestLoggingMiddleware
from hoploy.api.routes import admin, health, recommend
from hoploy.config.schema import ServingConfig


def create_app(config: ServingConfig) -> FastAPI:
    app = FastAPI(title="Hopwise-Serve", version="0.1.0")

    # Middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Routes
    app.include_router(health.router)
    app.include_router(recommend.router)
    app.include_router(admin.router)

    # Store config for route access
    app.state.config = config
    app.state.pipeline = None  # set during startup

    return app
