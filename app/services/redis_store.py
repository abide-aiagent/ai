"""
Redis session store — persist meditation session state in Redis
Sessions survive server restarts
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

from app.config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Redis client (initialised/closed in app lifespan)
# ──────────────────────────────────────────────
_redis: aioredis.Redis | None = None

SESSION_PREFIX = "abide:session:"
SESSION_TTL = 60 * 60 * 2  # 2 hours (meditation session lifetime)


async def init_redis() -> None:
    """Called at app startup — connect to Redis"""
    global _redis
    _redis = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    # Verify connection
    await _redis.ping()
    logger.info("Redis connected")


async def close_redis() -> None:
    """Called at app shutdown — close Redis connection"""
    global _redis
    if _redis:
        await _redis.close()
        _redis = None
        logger.info("Redis disconnected")


def get_redis() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialized — call init_redis() first")
    return _redis


# ──────────────────────────────────────────────
# Message serialization / deserialization (LangChain BaseMessage ⇄ dict)
# ──────────────────────────────────────────────
def _serialize_message(msg: BaseMessage) -> dict:
    """LangChain message → JSON-serializable dict"""
    return {
        "type": msg.type,  # "human", "ai", "system"
        "content": msg.content,
    }


def _deserialize_message(data: dict) -> BaseMessage:
    """dict → LangChain message"""
    msg_type = data["type"]
    content = data["content"]
    if msg_type == "human":
        return HumanMessage(content=content)
    elif msg_type == "ai":
        return AIMessage(content=content)
    elif msg_type == "system":
        return SystemMessage(content=content)
    else:
        return HumanMessage(content=content)


def _serialize_state(state: dict[str, Any]) -> str:
    """MeditationState → JSON string"""
    serializable = {}
    for key, value in state.items():
        if key == "messages":
            serializable[key] = [_serialize_message(m) for m in value]
        else:
            serializable[key] = value
    return json.dumps(serializable, ensure_ascii=False)


def _deserialize_state(data: str) -> dict[str, Any]:
    """JSON string → MeditationState dict"""
    parsed = json.loads(data)
    if "messages" in parsed:
        parsed["messages"] = [_deserialize_message(m) for m in parsed["messages"]]
    return parsed


# ──────────────────────────────────────────────
# Session CRUD
# ──────────────────────────────────────────────
async def save_session_state(session_id: str, state: dict[str, Any]) -> None:
    """Save session state to Redis"""
    r = get_redis()
    key = f"{SESSION_PREFIX}{session_id}"
    await r.set(key, _serialize_state(state), ex=SESSION_TTL)
    logger.debug(f"Session state saved: {session_id}")


async def load_session_state(session_id: str) -> dict[str, Any] | None:
    """Load session state from Redis — returns None if not found"""
    r = get_redis()
    key = f"{SESSION_PREFIX}{session_id}"
    data = await r.get(key)
    if data is None:
        return None
    return _deserialize_state(data)


async def delete_session_state(session_id: str) -> None:
    """Delete session state"""
    r = get_redis()
    key = f"{SESSION_PREFIX}{session_id}"
    await r.delete(key)
    logger.debug(f"Session state deleted: {session_id}")


async def extend_session_ttl(session_id: str) -> None:
    """Reset session TTL (called on conversation activity)"""
    r = get_redis()
    key = f"{SESSION_PREFIX}{session_id}"
    await r.expire(key, SESSION_TTL)
