"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from synth2surge.api.routes import router

    app = FastAPI(
        title="Synth2Surge API",
        description="Translate arbitrary VST synth patches into Surge XT patches.",
        version="0.1.0",
    )
    app.include_router(router)
    return app
