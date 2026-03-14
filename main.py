import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

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


_PUBLIC_PATHS = {"/", "/api/v1/agent/health"}


class InternalApiKeyMiddleware(BaseHTTPMiddleware):
    """Core 서버의 Internal API Key를 검증하는 미들웨어.
    모든 API 경로에 대해 X-Internal-Key 헤더를 확인합니다.
    (AI 서버는 Core 서버에서만 접근 가능)
    """

    async def dispatch(self, request: Request, call_next):
        from fastapi.responses import JSONResponse

        path = request.url.path
        # public paths skip key validation
        if path in _PUBLIC_PATHS:
            try:
                return await call_next(request)
            except Exception as e:
                logger.error(f"Unhandled error on public path {path}: {e}", exc_info=True)
                return JSONResponse(status_code=500, content={"detail": "Internal server error"})

        if not settings.internal_api_key:
            logger.error("INTERNAL_API_KEY not configured — rejecting all requests")
            return JSONResponse(status_code=503, content={"detail": "Service misconfigured"})

        key = request.headers.get("X-Internal-Key", "")
        if key != settings.internal_api_key:
            logger.warning(
                "Rejected request to %s — invalid X-Internal-Key from %s",
                path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(status_code=401, content={"detail": "Unauthorized: invalid internal API key"})

        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"Unhandled error on {path}: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def create_app() -> FastAPI:
    from fastapi import Request as FastAPIRequest
    from fastapi.responses import JSONResponse
    from fastapi.exceptions import RequestValidationError

    app = FastAPI(
        title="ABIDE AI Agent Server",
        description="Meditation agent, theology RAG, DeepLens analysis",
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: FastAPIRequest, exc: RequestValidationError):
        logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={"detail": "요청 형식이 올바르지 않습니다.", "errors": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: FastAPIRequest, exc: Exception):
        logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요."},
        )

    # CORS — 기본값은 Core 서버만 허용
    import os
    allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080")
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept", "X-Internal-Key"],
    )

    # Internal API key 검증 미들웨어
    app.add_middleware(InternalApiKeyMiddleware)

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
