"""
AI Agent Router — Meditation (Mate), Theology Search (Ask), DeepLens endpoints
SSE streaming support

모든 /api/v1/agent/* 엔드포인트는 Core 서버를 통해서만 접근 가능합니다.
(InternalApiKeyMiddleware에서 X-Internal-Key 헤더 검증)
"""

from __future__ import annotations

import copy
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.graph import (
    continue_meditation,
    continue_meditation_streaming,
    start_meditation,
    start_meditation_streaming,
)
# Note: Prompt templates imported from app/prompts/ (not included in public repo)
from app.prompts.prompts import ASK_SYSTEM_PROMPT, DEEP_LENS_PROMPT, MOOD_VERSES
from app.agents.state import MeditationState
from app.config import settings
from app.services.database import get_mood_verses_from_db, save_session_state_to_db, load_session_state_from_db, clear_session_snapshot
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
    ReportClassifyRequest,
    ReportClassifyResponse,
    TTSGenerateRequest,
    TTSGenerateResponse,
)
from app.services.database import (
    ensure_session_exists,
    get_deep_lens_cache,
    get_passage_text,
    save_ai_message,
    save_meditation_note,
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


def _normalize_application(value) -> list[str]:
    """application 필드 정규화 — LLM이 list 또는 string 모두 반환할 수 있음."""
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value:
        return [value]
    return []

# ──────────────────────────────────────────────
# Session state store (Redis-first, in-memory fallback)
# ──────────────────────────────────────────────
_session_states: dict[str, MeditationState] = {}  # In-memory fallback when Redis is unavailable


# 매 N turn마다 DB에 스냅샷 저장
DB_SNAPSHOT_INTERVAL = 3  # 3 turn마다 DB 저장

async def _save_state(session_id: str, state: MeditationState) -> None:
    """Redis 저장 + 일정 주기로 DB 스냅샷"""
    # 항상 Redis에 저장
    try:
        await save_session_state(session_id, state)
    except Exception:
        _session_states[session_id] = state

    # 3 turn마다 또는 세션 종료 시 DB에 스냅샷 저장
    # ⚠️ turn_count > 0 조건 추가: turn 0(세션 생성 직후)에 빈 스냅샷 저장 방지
    turn_count = state.get("turn_count", 0)
    is_final = state.get("end_confirmed", False)

    if (turn_count > 0 and turn_count % DB_SNAPSHOT_INTERVAL == 0) or is_final:
        try:
            await save_session_state_to_db(session_id, state)
            logger.info(f"세션 {session_id} DB 스냅샷 저장 완료 (turn={turn_count})")
        except Exception as e:
            logger.warning(f"DB 스냅샷 저장 실패 (무시됨): {e}")


async def _load_state(session_id: str) -> MeditationState | None:
    """Redis 조회 → 없으면 DB에서 복원 → 없으면 None"""
    # 1. Redis 조회
    try:
        state = await load_session_state(session_id)
        if state:
            await extend_session_ttl(session_id)
            return state
    except Exception:
        pass

    # 2. in-memory 폴백
    stored = _session_states.get(session_id)
    if stored:
        return copy.deepcopy(stored)

    # 3. DB에서 복원 (Redis miss + in-memory miss)
    try:
        state = await load_session_state_from_db(session_id)
        if state:
            logger.info(f"세션 {session_id} DB에서 복원 성공")
            # 복원된 상태를 Redis에 다시 캐싱
            await save_session_state(session_id, state)
            return state
    except Exception as e:
        logger.warning(f"DB 복원 실패: {e}")

    return None


async def _remove_state(session_id: str) -> None:
    """Delete session state"""
    try:
        await delete_session_state(session_id)
    except Exception:
        pass
    _session_states.pop(session_id, None)

    try:
        await clear_session_snapshot(session_id)
    except Exception as e:
        logger.warning(f"DB 세션 스냅샷 정리 실패: {e}")


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

            result: dict | None = None

            # 토큰 단위 스트리밍 — counselor/scribe 노드의 LLM 출력을 즉시 전달
            async for item in start_meditation_streaming(
                user_id=req.user_id,
                session_id=req.session_id,
                verse_ref=req.verse_ref,
                verse_refs=req.verse_refs,
                mood=req.mood or "",
                session_type=req.session_type,
                initial_query=req.initial_query,
            ):
                if item["type"] == "token":
                    yield _sse_event("message", {"chunk": item["content"]})
                elif item["type"] == "result":
                    result = item["data"]

            if result is None:
                yield _sse_event("error", {"message": "묵상 응답을 생성하지 못했습니다."})
                return

            # 비스트리밍 노드(confirm_end, wrap_up 등)의 응답을 전체 텍스트로 전송
            if not result.get("was_streamed") and result.get("content"):
                yield _sse_event("message", {"content": result["content"]})

            # Save session state (Redis)
            await _save_state(req.session_id, result["state"])

            # Send thinking events (CoT monitoring)
            for entry in result.get("thinking_log", []):
                yield _sse_event("thinking", entry)

            # Send referenced verse events
            for verse in result.get("referenced_verses", []):
                yield _sse_event("verse_highlight", verse)

            elapsed = int((time.time() - start_time) * 1000)

            # DB 저장 (done 이벤트 전에 완료)
            await save_ai_message(
                session_id=req.session_id,
                role="assistant",
                content=result["content"],
                referenced_verses=result.get("referenced_verses"),
                latency_ms=elapsed,
            )

            if result.get("meditation_note"):
                await save_meditation_note(req.session_id, result["meditation_note"])

            # done 이벤트 (DB 저장 완료 후)
            yield _sse_event(
                "done",
                {
                    "session_id": req.session_id,
                    "meditation_depth": result["meditation_depth"],
                    "turn_count": result["turn_count"],
                    "latency_ms": elapsed,
                },
            )

        except Exception as e:
            logger.error(f"Meditation start error: {e}", exc_info=True)
            yield _sse_event("error", {"message": "묵상 시작 중 오류가 발생했습니다. 다시 시도해주세요."})

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

            result: dict | None = None

            # 토큰 단위 스트리밍 — counselor/scribe 노드의 LLM 출력을 즉시 전달
            async for item in continue_meditation_streaming(
                session_state=session_state,
                user_message=req.user_message,
            ):
                if item["type"] == "token":
                    yield _sse_event("message", {"chunk": item["content"]})
                elif item["type"] == "result":
                    result = item["data"]

            if result is None:
                yield _sse_event("error", {"message": "묵상 응답을 생성하지 못했습니다."})
                return

            # 비스트리밍 노드(confirm_end, wrap_up 등)의 응답을 전체 텍스트로 전송
            if not result.get("was_streamed") and result.get("content"):
                yield _sse_event("message", {"content": result["content"]})

            # Update session state (Redis)
            await _save_state(req.session_id, result["state"])

            # Send thinking events
            for entry in result.get("thinking_log", []):
                yield _sse_event("thinking", entry)

            # Referenced verses
            for verse in result.get("referenced_verses", []):
                yield _sse_event("verse_highlight", verse)

            elapsed = int((time.time() - start_time) * 1000)

            # DB 저장 (done 이벤트 전에 완료)
            await save_ai_message(
                session_id=req.session_id,
                role="assistant",
                content=result["content"],
                referenced_verses=result.get("referenced_verses"),
                latency_ms=elapsed,
            )

            if result.get("meditation_note"):
                await save_meditation_note(req.session_id, result["meditation_note"])

            done_data = {
                "session_id": req.session_id,
                "meditation_depth": result["meditation_depth"],
                "turn_count": result["turn_count"],
                "is_final": result["is_final"],
                "latency_ms": elapsed,
            }
            if result.get("meditation_note"):
                done_data["meditation_note"] = result["meditation_note"]

            yield _sse_event("done", done_data)

            if result["is_final"]:
                await _remove_state(req.session_id)

        except Exception as e:
            logger.error(f"Meditation chat error: {e}", exc_info=True)
            yield _sse_event("error", {"message": "묵상 진행 중 오류가 발생했습니다. 다시 시도해주세요."})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Non-SSE version (debugging/testing)
@router.post("/meditation/start/sync", response_model=MeditationResponse)
async def meditation_start_sync(req: MeditationStartRequest):
    """Start meditation session (sync response — for debugging)"""
    try:
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
            session_type=req.session_type,
            initial_query=req.initial_query,
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Meditation start sync error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="묵상 시작 중 오류가 발생했습니다.")


@router.post("/meditation/chat/sync", response_model=MeditationResponse)
async def meditation_chat_sync(req: MeditationChatRequest):
    """Meditation chat turn (sync response — for debugging)"""
    session_state = await _load_state(req.session_id)
    if not session_state:
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Meditation chat sync error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="묵상 진행 중 오류가 발생했습니다.")


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
            yield _sse_event("error", {"message": "질문 처리 중 오류가 발생했습니다. 다시 시도해주세요."})

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
    try:
        # Cache check
        if not req.force_refresh:
            try:
                cached = await get_deep_lens_cache(req.verse_ref)
            except Exception as e:
                logger.warning(f"DeepLens cache read failed (proceeding without cache): {e}")
                cached = None

            if cached:
                return DeepLensResponse(
                    verse_ref=req.verse_ref,
                    context_guide=cached["context_guide"],
                    interpretation=cached["interpretation"],
                    application=_normalize_application(cached["application"]),
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
            logger.warning(f"Failed to fetch verse text for {req.verse_ref}: {e}")

        # RAG search (has internal try-except, returns [] on failure)
        rag_results = await search_theology(verse_text or req.verse_ref, limit=4)
        rag_context = format_rag_context(rag_results)

        # LLM analysis
        try:
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
            raw_content = resp.content
        except Exception as e:
            logger.error(f"DeepLens LLM call failed for {req.verse_ref}: {e}", exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="AI 분석 서비스에 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            )

        # Parse LLM JSON response
        try:
            # Strip markdown code blocks if present
            content = raw_content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                )
            result = json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"DeepLens JSON parse failed for {req.verse_ref}, using raw content")
            result = {
                "context_guide": raw_content,
                "interpretation": "",
                "application": "",
                "cross_references": [],
            }

        # Cache save (non-critical — failure does not affect response)
        try:
            await save_deep_lens_cache(
                verse_ref=req.verse_ref,
                context_guide=result.get("context_guide", ""),
                interpretation=result.get("interpretation", ""),
                application=result.get("application", ""),
                cross_references=result.get("cross_references"),
            )
        except Exception as e:
            logger.warning(f"DeepLens cache save failed (non-critical): {e}")

        return DeepLensResponse(
            verse_ref=req.verse_ref,
            verse_text=verse_text,
            context_guide=result.get("context_guide", ""),
            interpretation=result.get("interpretation", ""),
            application=_normalize_application(result.get("application")),
            cross_references=result.get("cross_references", []),
            cached=False,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DeepLens unexpected error for {req.verse_ref}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="딥렌즈 분석 중 예상치 못한 오류가 발생했습니다.",
        )


# ══════════════════════════════════════════════
# Compass — Mood Check-in & Recommended Verses
# ══════════════════════════════════════════════
@router.post("/compass/check-in", response_model=MoodCheckInResponse)
async def mood_check_in(req: MoodCheckInRequest):
    """Mood check-in — returns mood-based verse recommendations"""
    try:
        mood = req.mood.lower()

        # DB에서 먼저 조회, 실패 시 하드코딩 폴백
        verses = await get_mood_verses_from_db(mood)
        if not verses:
            verses = await get_mood_verses_from_db("sad")  # fallback mood
        if not verses:
            verses = MOOD_VERSES.get(mood, MOOD_VERSES.get("sad", []))  # final fallback

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
    except Exception as e:
        logger.error(f"Compass check-in error (mood={req.mood}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="감정 체크인 처리 중 오류가 발생했습니다.")

# ══════════════════════════════════════════════
# Report Classification (L6)
# ══════════════════════════════════════════════
@router.post("/classify-report", response_model=ReportClassifyResponse)
async def classify_report(req: ReportClassifyRequest):
    """Auto-classify user reports using LLM (best-effort; returns safe default on failure)"""
    prompt = f"""
    다음 신고 내용을 분석하여 심각도(severity)와 권장 조치(suggested_action)를 분류하세요.
    신고 사유: {req.report_reason}
    대상 내용: {req.target_content or '없음'}

    결과는 반드시 JSON 형식으로 반환하세요:
    {{
        "severity": "low" | "medium" | "high" | "critical",
        "suggested_action": "ignore" | "review" | "hide" | "ban",
        "confidence": 0.0 ~ 1.0
    }}
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=0.0,
            google_api_key=settings.gemini_api_key,
        )
        msg = await llm.ainvoke([HumanMessage(content=prompt)])

        # Parse JSON from markdown block if necessary
        content = msg.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        data = json.loads(content)
        return ReportClassifyResponse(
            severity=data.get("severity", "medium"),
            suggested_action=data.get("suggested_action", "review"),
            confidence=data.get("confidence", 0.5),
        )
    except Exception as e:
        logger.error(f"Failed to classify report: {e}", exc_info=True)
        return ReportClassifyResponse(
            severity="medium",
            suggested_action="review",
            confidence=0.0,
        )

# ══════════════════════════════════════════════
# TTS Generation Stub (L8)
# ══════════════════════════════════════════════
@router.post("/tts/generate", response_model=TTSGenerateResponse)
async def generate_tts(req: TTSGenerateRequest):
    """Stub endpoint for generating TTS meditation guides"""
    # TODO: Implement actual TTS generation (e.g. Google Cloud TTS, ElevenLabs, etc.)
    return TTSGenerateResponse(
        audio_url="https://example.com/audio/stub_meditation_guide.mp3",
        duration=15.5
    )
