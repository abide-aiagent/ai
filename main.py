import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import agent
from app.services.database import close_pool, init_pool
from app.services.redis_store import close_redis, init_redis

# Logging setup
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage resources on app startup/shutdown"""
    # Startup: Initialize DB pool + Redis
    try:
        await init_pool()
        logger.info("Database pool initialized")
    except Exception as e:
        logger.warning(f"Database pool init failed (running without DB): {e}")

    try:
        await init_redis()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis init failed (falling back to in-memory sessions): {e}")

    yield

    # Shutdown: Close Redis + DB pool
    try:
        await close_redis()
    except Exception:
        pass
    try:
        await close_pool()
        logger.info("Database pool closed")
    except Exception:
        pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="ABIDE AI Agent Server",
        description="Meditation agent, theology RAG, DeepLens analysis",
        version="0.2.0",
        lifespan=lifespan,
    )

    # CORS
    import os
    allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
    allowed_origins = ["*"] if allowed_origins_env == "*" else [o.strip() for o in allowed_origins_env.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register router
    app.include_router(agent.router)

    @app.get("/")
    async def root():
        return {"service": "abide_ai", "status": "running", "version": "0.2.0"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.ai_server_port,
        reload=True,
    )
