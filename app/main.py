from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.settings import get_settings

from app.db import close_pool, init_pool, open_pool
from app.db.migrator import run_migrations

from app.services.graph import build_agent_graph, open_checkpointer

from app.api.routers import health, threads

from app.core.logging import configure_logging

def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        init_pool(
            settings.database_url,
            settings.db_pool_min_size,
            settings.db_pool_max_size,
        )
        await open_pool()
        await run_migrations()

        checkpointer_stack, checkpointer = await open_checkpointer(settings.database_url)
        await checkpointer.setup()

        app.state.checkpointer_stack = checkpointer_stack
        app.state.checkpointer = checkpointer
        app.state.graph = build_agent_graph(checkpointer)

        try:
            yield
        finally:
            # Shutdown
            await close_pool()

            stack = getattr(app.state, "checkpointer_stack", None)
            if stack is not None:
                await stack.aclose()

            app.state.checkpointer_stack = None
            app.state.checkpointer = None
            app.state.graph = None

    app = FastAPI(title=settings.title, version=settings.version, lifespan=lifespan)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=settings.allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(threads.router)

    return app

app = create_app()


