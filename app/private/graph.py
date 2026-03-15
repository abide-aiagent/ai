"""
AI Agent Router — Meditation (Mate), Theology Search (Ask), DeepLens endpoints
SSE streaming support
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.graph import continue_meditation, start_meditation
# Note: Prompt templates imported from app/prompts/ (not included in public repo)
from app.prompts.prompts import ASK_SYSTEM_PROMPT, DEEP_LENS_PROMPT, MOOD_VERSES
from app.agents.state import MeditationState
from app.config import settings
from app.models.schemas import (
    AskRequest,
    AskResponse,
    DeepLensRequest,
    DeepLensResponse,
    MeditationChatRequest,
    MeditationResponse,
    MeditationStartRequest,
    MoodCheckInRequest,
    MoodCheckInResponse,
)
from app.services.database import (
    ensure_session_exists,
    get_deep_lens_cache,
    get_passage_text,
    save_ai_message,
    save_deep_lens_cache,
)
from app.services.rag import format_rag_context, search_theology
from app.services.redis_store import (
    delete_session_state,
    extend_session_ttl,
    load_session_state,
    save_session_state,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/agent", tags=["AI Agent"])

# ──────────────────────────────────────────────
# Session state store (Redis-first, in-memory fallback)
# ──────────────────────────────────────────────
_session_states: dict[str, MeditationState] = {}  # In-memory fallback when Redis is unavailable


async def _save_state(session_id: str, state: MeditationState) -> None:
    """Save session state (Redis-first, in-memory fallback on failure)"""
    try:
        await save_session_state(session_id, state)
    except Exception:
        _session_states[session_id] = state


async def _load_state(session_id: str) -> MeditationState | None:
    """Load session state (Redis-first, in-memory fallback on failure)"""
    try:
        state = await load_session_state(session_id)
        if state:
            await extend_session_ttl(session_id)
            return state
    except Exception:
        pass
    return _session_states.get(session_id)


async def _remove_state(session_id: str) -> None:
    """Delete session state"""
    try:
        await delete_session_state(session_id)
    except Exception:
        pass
    _session_states.pop(session_id, None)


# ──────────────────────────────────────────────
# SSE Event Format Helper
# ──────────────────────────────────────────────
def _sse_event(event: str, data: dict | str) -> str:
    """Build an SSE event string"""
    if isinstance(data, dict):
        data = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


# ══════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════
@router.get("/health")
async def health():
    return {"status": "ok", "service": "abide_ai", "version": "0.1.0"}


# ══════════════════════════════════════════════
# Meditation (Mate)
# ══════════════════════════════════════════════
@router.post("/meditation/start")
async def meditation_start(req: MeditationStartRequest):
    """
    Start a meditation session — initialise LangGraph workflow.
    Responds via SSE streaming.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            start_time = time.time()

            # Ensure session row exists (needed when AI server is called directly)
            await ensure_session_exists(
                session_id=req.session_id,
                verse_ref=req.verse_ref,
            )

            result = await start_meditation(
                user_id=req.user_id,
                session_id=req.session_id,
                verse_ref=req.verse_ref,
                verse_refs=req.verse_refs,
                mood=req.mood or "",
            )

            # Save session state (Redis)
            await _save_state(req.session_id, result["state"])

            # Send thinking events (CoT monitoring)
            for entry in result.get("thinking_log", []):
                yield _sse_event("thinking", entry)

            # Send message event
            yield _sse_event("message", {"chunk": result["content"]})

            # Send referenced verse events
            for verse in result.get("referenced_verses", []):
                yield _sse_event("verse_highlight", verse)

            elapsed = int((time.time() - start_time) * 1000)
            yield _sse_event(
                "done",
                {
                    "session_id": req.session_id,
                    "meditation_depth": result["meditation_depth"],
                    "turn_count": result["turn_count"],
                    "latency_ms": elapsed,
                },
            )

            # Save AI message to DB
            await save_ai_message(
                session_id=req.session_id,
                role="assistant",
                content=result["content"],
                referenced_verses=result.get("referenced_verses"),
                latency_ms=elapsed,
            )

        except Exception as e:
            logger.error(f"Meditation start error: {e}", exc_info=True)
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/meditation/chat")
async def meditation_chat(req: MeditationChatRequest):
    """
    Meditation chat turn — receives user message and streams agent response via SSE.
    """
    session_state = await _load_state(req.session_id)
    if not session_state:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new meditation.")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            start_time = time.time()

            # Save user message to DB
            await save_ai_message(
                session_id=req.session_id,
                role="user",
                content=req.user_message,
            )

            result = await continue_meditation(
                session_state=session_state,
                user_message=req.user_message,
            )

            # Update session state (Redis)
            await _save_state(req.session_id, result["state"])

            # Send thinking events
            for entry in result.get("thinking_log", []):
                yield _sse_event("thinking", entry)

            # Send message event
            yield _sse_event("message", {"chunk": result["content"]})

            # Referenced verses
            for verse in result.get("referenced_verses", []):
                yield _sse_event("verse_highlight", verse)

            elapsed = int((time.time() - start_time) * 1000)

            done_data = {
                "session_id": req.session_id,
                "meditation_depth": result["meditation_depth"],
                "turn_count": result["turn_count"],
                "is_final": result["is_final"],
                "latency_ms": elapsed,
            }

            # Include meditation note if generated
            if result.get("meditation_note"):
                done_data["meditation_note"] = result["meditation_note"]

            yield _sse_event("done", done_data)

            # Save AI message to DB
            await save_ai_message(
                session_id=req.session_id,
                role="assistant",
                content=result["content"],
                referenced_verses=result.get("referenced_verses"),
                latency_ms=elapsed,
            )

            # Clean up session state on meditation completion
            if result["is_final"]:
                await _remove_state(req.session_id)

        except Exception as e:
            logger.error(f"Meditation chat error: {e}", exc_info=True)
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Non-SSE version (debugging/testing)
@router.post("/meditation/start/sync", response_model=MeditationResponse)
async def meditation_start_sync(req: MeditationStartRequest):
    """Start meditation session (sync response — for debugging)"""
    result = await start_meditation(
        user_id=req.user_id,
        session_id=req.session_id,
        verse_ref=req.verse_ref,
        verse_refs=req.verse_refs,
        mood=req.mood or "",
    )
    await _save_state(req.session_id, result["state"])

    return MeditationResponse(
        session_id=req.session_id,
        content=result["content"],
        thinking_log=result.get("thinking_log", []),
        meditation_depth=result["meditation_depth"],
        turn_count=result["turn_count"],
        referenced_verses=result.get("referenced_verses", []),
        is_final=result["is_final"],
        meditation_note=result.get("meditation_note"),
    )


@router.post("/meditation/chat/sync", response_model=MeditationResponse)
async def meditation_chat_sync(req: MeditationChatRequest):
    """Meditation chat turn (sync response — for debugging)"""
    session_state = await _load_state(req.session_id)
    if not session_state:
        raise HTTPException(status_code=404, detail="Session not found.")

    result = await continue_meditation(
        session_state=session_state,
        user_message=req.user_message,
    )
    await _save_state(req.session_id, result["state"])

    if result["is_final"]:
        await _remove_state(req.session_id)

    return MeditationResponse(
        session_id=req.session_id,
        content=result["content"],
        thinking_log=result.get("thinking_log", []),
        meditation_depth=result["meditation_depth"],
        turn_count=result["turn_count"],
        referenced_verses=result.get("referenced_verses", []),
        is_final=result["is_final"],
        meditation_note=result.get("meditation_note"),
    )


# ══════════════════════════════════════════════
# Ask — Theology Search
# ══════════════════════════════════════════════
@router.post("/ask")
async def theology_search(req: AskRequest):
    """Theology question search (RAG) — SSE streaming"""

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # RAG search
            rag_results = await search_theology(req.query, limit=5)
            rag_context = format_rag_context(rag_results)

            # Send source events
            for r in rag_results:
                if r.get("source_title"):
                    yield _sse_event("source", {
                        "title": r["source_title"],
                        "type": r.get("source_type", "unknown"),
                    })

            # Generate answer with LLM
            llm = ChatGoogleGenerativeAI(
                model=settings.llm_model,
                temperature=0.5,
                google_api_key=settings.gemini_api_key,
                streaming=True,
            )

            system = ASK_SYSTEM_PROMPT.format(
                rag_context=rag_context or "(참조 자료 없음 — 성경 본문 기반으로 답변)",
            )

            async for chunk in llm.astream([
                SystemMessage(content=system),
                HumanMessage(content=req.query),
            ]):
                if chunk.content:
                    yield _sse_event("message", {"chunk": chunk.content})

            yield _sse_event("done", {"session_id": req.session_id})

        except Exception as e:
            logger.error(f"Ask error: {e}", exc_info=True)
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════
# DeepLens — Deep Analysis
# ══════════════════════════════════════════════
@router.post("/deep-lens", response_model=DeepLensResponse)
async def deep_lens_analyze(req: DeepLensRequest):
    """DeepLens deep analysis — cache-first, AI-generated on miss"""

    # Cache check
    if not req.force_refresh:
        cached = await get_deep_lens_cache(req.verse_ref)
        if cached:
            return DeepLensResponse(
                verse_ref=req.verse_ref,
                context_guide=cached["context_guide"],
                interpretation=cached["interpretation"],
                application=cached["application"],
                cross_references=cached.get("cross_references") or [],
                cached=True,
            )

    # Fetch Bible verse text
    verse_text = ""
    try:
        parts = req.verse_ref.split(":")
        if len(parts) >= 4:
            version, book, chapter, verse = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            verse_text = await get_passage_text(version, book, chapter, verse)
    except Exception as e:
        logger.warning(f"Failed to fetch verse text: {e}")

    # RAG search
    rag_results = await search_theology(verse_text or req.verse_ref, limit=4)
    rag_context = format_rag_context(rag_results)

    # LLM analysis
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.5,
        google_api_key=settings.gemini_api_key,
    )

    prompt = DEEP_LENS_PROMPT.format(
        verse_ref=req.verse_ref,
        verse_text=verse_text or "(본문 로딩 실패)",
        rag_context=rag_context or "(참조 자료 없음)",
    )

    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        result = json.loads(resp.content)
    except json.JSONDecodeError:
        result = {
            "context_guide": resp.content,
            "interpretation": "",
            "application": "",
            "cross_references": [],
        }

    # Cache save
    await save_deep_lens_cache(
        verse_ref=req.verse_ref,
        context_guide=result.get("context_guide", ""),
        interpretation=result.get("interpretation", ""),
        application=result.get("application", ""),
        cross_references=result.get("cross_references"),
    )

    return DeepLensResponse(
        verse_ref=req.verse_ref,
        verse_text=verse_text,
        context_guide=result.get("context_guide", ""),
        interpretation=result.get("interpretation", ""),
        application=result.get("application", ""),
        cross_references=result.get("cross_references", []),
        cached=False,
    )


# ══════════════════════════════════════════════
# Compass — Mood Check-in & Recommended Verses
# ══════════════════════════════════════════════
@router.post("/compass/check-in", response_model=MoodCheckInResponse)
async def mood_check_in(req: MoodCheckInRequest):
    """Mood check-in — returns mood-based verse recommendations"""
    mood = req.mood.lower()
    verses = MOOD_VERSES.get(mood, MOOD_VERSES.get("sad", []))

    mood_messages = {
        "joy": "기쁜 날이네요! 주 안에서 함께 기뻐해요. 🎉",
        "anxious": "하나님의 평안이 함께하시길 기도합니다. 🙏",
        "angry": "마음이 격앙된 순간에도 하나님은 함께 하세요.",
        "tired": "쉬어가도 괜찮아요. 하나님이 새 힘을 주실 거예요. 💚",
        "grateful": "감사하는 마음, 정말 아름다워요! 🌿",
        "sad": "슬플 때 하나님은 더 가까이 계세요. 위로가 함께하길 빕니다. 💙",
    }

    return MoodCheckInResponse(
        message=mood_messages.get(mood, "하나님의 은혜가 함께하시길 빕니다."),
        recommended_verses=verses,
    )
