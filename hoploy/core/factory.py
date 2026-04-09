from __future__ import annotations

import importlib
import inspect as _inspect
import pathlib
from contextlib import asynccontextmanager
from typing import Dict, Type

from fastapi import APIRouter, FastAPI, Request as FastAPIRequest, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from omegaconf import OmegaConf

from hoploy.core.config import Config


# ---------------------------------------------------------------------------
# Schema decorators – used by plugin schema modules to mark request/response
# ---------------------------------------------------------------------------

def Request(name: str):
    """Decorator to mark a Pydantic model as the request schema for *name*.

    :param name: The endpoint name this schema belongs to.
    :type name: str
    """
    def decorator(cls):
        cls._endpoint_name = name
        cls._endpoint_role = "request"
        return cls
    return decorator


def Response(name: str):
    """Decorator to mark a Pydantic model as the response schema for *name*.

    :param name: The endpoint name this schema belongs to.
    :type name: str
    """
    def decorator(cls):
        cls._endpoint_name = name
        cls._endpoint_role = "response"
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Schema collection helpers
# ---------------------------------------------------------------------------

def _collect_schemas(schema_module):
    """Scan *schema_module* for ``@Request`` / ``@Response``-decorated classes.

    :param schema_module: An imported Python module to inspect.
    :returns: ``{endpoint_name: {"request": cls, "response": cls}}``.
    :rtype: dict[str, dict[str, type]]
    """
    endpoints: Dict[str, Dict[str, Type]] = {}
    for _attr_name, obj in _inspect.getmembers(schema_module, _inspect.isclass):
        ep = getattr(obj, "_endpoint_name", None)
        role = getattr(obj, "_endpoint_role", None)
        if ep and role in ("request", "response"):
            endpoints.setdefault(ep, {})[role] = obj
    return endpoints


# ---------------------------------------------------------------------------
# Router builder
# ---------------------------------------------------------------------------

def _resolve_handler(handler_spec, service):
    """Resolve a handler specification to a callable.

    * ``"run"`` resolves to ``service.run`` (pipeline method).
    * ``"component_name.method"`` resolves via
      :meth:`~hoploy.core.pipeline.Pipe.get_component` so that the
      method is bound to the live instance, not the class.

    :param handler_spec: Dotted or plain handler name.
    :type handler_spec: str
    :param service: The active :class:`~hoploy.core.pipeline.Pipe` instance.
    :returns: The resolved callable.
    :raises ValueError: If the handler or method cannot be found.
    """
    if "." in handler_spec:
        component_name, method_name = handler_spec.rsplit(".", 1)
        instance = service.get_component(component_name)
        method = getattr(instance, method_name, None)
        if method is None:
            raise ValueError(
                f"Component '{component_name}' has no method '{method_name}'"
            )
        return method
    # Simple name → pipeline method
    method = getattr(service, handler_spec, None)
    if method is None:
        raise ValueError(f"Pipeline has no method '{handler_spec}'")
    return method


def _build_router(cfg):
    """Build a FastAPI APIRouter from the plugin schema config.

    The ``schema`` block in each plugin declares endpoints::

        schema:
          module: autism_schema
          get:
            info: autism_model.info
          post:
            recommend: run
            search: autism_model.search

    Handler values can be:
    - ``"run"`` — route through the full pipeline (Pipe.run)
    - ``"component.method"`` — call method directly on a registered component
    """
    router = APIRouter()

    def _get_service(request: FastAPIRequest):
        service = getattr(request.app.state, "service", None)
        if not service or not service.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service warming up",
            )
        return service

    def _make_get(handler_spec, field_name, resp_cls):
        async def handler(request: FastAPIRequest, service=Depends(_get_service), **kwargs):
            value = kwargs.get(field_name) or request.path_params.get(field_name)
            resolved = _resolve_handler(handler_spec, service)
            req_config = Config(_raw=OmegaConf.create({field_name: value}))
            result = resolved(req_config)
            if _inspect.isawaitable(result):
                result = await result
            if result is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Not found",
                )
            return resp_cls.model_validate(result)

        old_sig = _inspect.signature(handler)
        params = [
            p for p in old_sig.parameters.values()
            if p.kind != _inspect.Parameter.VAR_KEYWORD
        ]
        path_param = _inspect.Parameter(
            field_name,
            _inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=str,
        )
        handler.__signature__ = old_sig.replace(parameters=[path_param] + params)
        handler.__name__ = f"get_{handler_spec.replace('.', '_')}"
        return handler

    def _make_post(handler_spec, req_cls, resp_cls):
        async def handler(body: req_cls, service=Depends(_get_service)):
            resolved = _resolve_handler(handler_spec, service)
            req_config = Config(_raw=OmegaConf.create(body.model_dump()))
            result = resolved(req_config)
            if _inspect.isawaitable(result):
                result = await result
            if result is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Not found",
                )
            return resp_cls.model_validate(result)

        handler.__annotations__["body"] = req_cls
        handler.__annotations__["return"] = resp_cls
        handler.__name__ = f"post_{handler_spec.replace('.', '_')}"
        return handler

    for _plugin_name, plugin_cfg in cfg.plugin.raw.items():
        schema_cfg = plugin_cfg.get("schema")
        if not schema_cfg or isinstance(schema_cfg, str):
            continue

        module_name = schema_cfg.get("module")
        if not module_name:
            continue

        plugin_path = pathlib.Path(
            plugin_cfg.get("path", f"plugins/{_plugin_name}")
        )
        schema_module = importlib.import_module(
            f"{plugin_path.name}.{module_name}"
        )
        schemas = _collect_schemas(schema_module)

        # --- GET endpoints ---
        for ep_name, handler_spec in (schema_cfg.get("get") or {}).items():
            ep = schemas.get(ep_name)
            if not ep or "request" not in ep or "response" not in ep:
                raise ValueError(
                    f"Missing @Request/@Response for GET endpoint '{ep_name}' "
                    f"in module '{schema_module.__name__}'"
                )
            req_cls, resp_cls = ep["request"], ep["response"]
            field_name = next(iter(req_cls.model_fields))
            router.add_api_route(
                f"/{ep_name}/{{{field_name}}}",
                _make_get(handler_spec, field_name, resp_cls),
                methods=["GET"],
                response_model=resp_cls,
                tags=[ep_name],
            )

        # --- POST endpoints ---
        for ep_name, handler_spec in (schema_cfg.get("post") or {}).items():
            ep = schemas.get(ep_name)
            if not ep or "request" not in ep or "response" not in ep:
                raise ValueError(
                    f"Missing @Request/@Response for POST endpoint '{ep_name}' "
                    f"in module '{schema_module.__name__}'"
                )
            req_cls, resp_cls = ep["request"], ep["response"]
            router.add_api_route(
                f"/{ep_name}",
                _make_post(handler_spec, req_cls, resp_cls),
                methods=["POST"],
                response_model=resp_cls,
                tags=[ep_name],
            )

    return router


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def factory(pipeline, cfg):
    """Create and return a fully configured :class:`FastAPI` application.

    :param pipeline: An already-initialised pipeline instance.
    :type pipeline: ~hoploy.core.pipeline.Pipe
    :param cfg: The full application config (used to discover plugin schemas).
    :type cfg: ~hoploy.core.config.Config
    :returns: The configured FastAPI app.
    :rtype: FastAPI
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = pipeline
        try:
            yield
        finally:
            await pipeline.shutdown()

    app = FastAPI(
        title="Hopwise-Serve",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/")
    async def root():
        return RedirectResponse(url="/docs")

    @app.get("/health", tags=["health"])
    def health() -> dict:
        return {"status": "ok"}

    # Auto-generate routes from plugin schema config
    app.include_router(_build_router(cfg))

    return app