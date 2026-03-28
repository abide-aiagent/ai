# Note: Some implementation details are not included in the public repository.
# See function docstrings for descriptions.
"""
Meditation Agent Node Implementations (LangGraph Nodes)

7 nodes: Supervisor, Planner, Counselor, Observer, Scribe, ConfirmEnd, WrapUp

최적화 내역:
- Counselor: MATE_SYSTEM_PROMPT + COUNSELOR_PROMPT 2개 → COUNSELOR_SYSTEM_PROMPT 1개로 통합
  (scripture/emotion 중복 전달 제거, 플레이스홀더 미사용 버그 수정)
- Planner: key_rag_insights 압축 필드 추출 → Counselor에는 전체 RAG 대신 핵심만 전달
- ConfirmEnd/WrapUp: LLM 호출 제거 → 템플릿 기반으로 대체 (세션당 LLM 2회 절약)
- Observer: 프롬프트 단축, 불필요한 HumanMessage 제거
- 모든 노드에서 scripture_text 중복 전달 제거
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.prompts.prompts import (
    COUNSELOR_SYSTEM_PROMPT,
    OBSERVER_PROMPT,
    PLANNER_PROMPT,
    SCRIBE_PROMPT,
    WRAP_UP_MESSAGE,
    get_confirm_end_message,
)
from app.agents.state import MeditationState, ThinkingEntry
from app.config import settings
from app.services.rag import format_rag_context, search_theology

logger = logging.getLogger(__name__)


def _get_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """Create an LLM instance."""
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=temperature,
        google_api_key=settings.gemini_api_key,
    )


def _extract_text_content(content) -> str:
    """LLM 응답의 content를 문자열로 변환 (Gemini multipart 대응)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response (handles markdown code blocks)."""
    text = _extract_text_content(text)
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip() == "```" and in_block:
                break
            elif in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"JSON parse failed, returning raw text: {text[:200]}")
        return {"raw": text}


# ──────────────────────────────────────────────
# 1. Supervisor Node (pure code-based routing — no LLM)
# ──────────────────────────────────────────────
async def supervisor_node(state: MeditationState) -> dict:
    """Controls the overall meditation flow. Pure code-based routing — no LLM call."""
    turn_count = state.get("turn_count", 0)
    depth = state.get("meditation_depth", 0)
    end_confirmed = state.get("end_confirmed", False)
    note_requested = state.get("note_requested")

    messages = state.get("messages", [])
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content.lower()
            break

    # ── Keyword sets ──
    end_keywords = {"정리", "마무리", "기도문", "끝", "종료", "노트"}
    confirm_yes_keywords = {"네", "응", "끝낼게", "종료", "끝", "그래", "마무리", "좋아"}
    confirm_no_keywords = {"아니", "계속", "더", "아직", "이어서", "안 끝"}
    note_accept_keywords = {"노트", "만들어", "네", "좋아", "응", "부탁", "작성", "만들",
                            "좋습니다", "알겠어요", "할게요", "예", "ㅇㅇ", "감사", "해줘", "해주세요", "부탁해"}
    note_reject_keywords = {"아니", "괜찮", "안 만들", "됐어", "필요없", "다음에",
                            "안해", "안 해", "노트 없이", "그냥", "넘어가"}

    user_wants_end = any(kw in last_user_msg for kw in end_keywords)

    last_node = state.get("last_executed_node", "")
    was_confirm_end = last_node == "confirm_end"
    was_wrap_up = last_node == "wrap_up"

    # ── (A) After confirm_end: user responds yes/no ──
    if was_confirm_end and not end_confirmed:
        wants_end = any(kw in last_user_msg for kw in confirm_yes_keywords)
        wants_continue = any(kw in last_user_msg for kw in confirm_no_keywords)
        if wants_end:
            reasoning = "사용자가 묵상 종료 확정"
            next_step = "wrap_up"
            return {
                "next_step": next_step,
                "thinking_log": [ThinkingEntry(node="supervisor", reasoning=reasoning, decision=f"→ {next_step}")],
                "turn_count": turn_count,
                "end_confirmed": True,
            }
        else:
            reasoning = "사용자가 묵상 계속" if wants_continue else "응답 모호, 계속 진행"
            next_step = "observer"
            return {
                "next_step": next_step,
                "thinking_log": [ThinkingEntry(node="supervisor", reasoning=reasoning, decision=f"→ {next_step}")],
                "turn_count": turn_count,
                "end_confirmed": False,
            }

    # ── (B) After wrap_up: user responds about note ──
    wrap_up_retry_count = state.get("wrap_up_retry_count", 0)
    if (was_wrap_up or end_confirmed) and note_requested is None and turn_count > 0:
        wants_note = any(kw in last_user_msg for kw in note_accept_keywords)
        rejects_note = any(kw in last_user_msg for kw in note_reject_keywords)
        if wants_note:
            next_step = "scribe"
            reasoning = "노트 생성 요청"
            return {
                "next_step": next_step,
                "thinking_log": [ThinkingEntry(node="supervisor", reasoning=reasoning, decision=f"→ {next_step}")],
                "turn_count": turn_count,
                "note_requested": True,
                "wrap_up_retry_count": 0,
            }
        elif rejects_note:
            next_step = "scribe"
            reasoning = "노트 없이 마무리"
            return {
                "next_step": next_step,
                "thinking_log": [ThinkingEntry(node="supervisor", reasoning=reasoning, decision=f"→ {next_step}")],
                "turn_count": turn_count,
                "note_requested": False,
                "wrap_up_retry_count": 0,
            }
        elif wrap_up_retry_count >= 1:
            # 재질문 후에도 응답 불명확 → 기본값으로 노트 생성
            next_step = "scribe"
            reasoning = "노트 여부 응답 불명확 (재질문 초과) — 기본 노트 생성"
            return {
                "next_step": next_step,
                "thinking_log": [ThinkingEntry(node="supervisor", reasoning=reasoning, decision=f"→ {next_step}")],
                "turn_count": turn_count,
                "note_requested": True,
                "wrap_up_retry_count": 0,
            }
        else:
            next_step = "wrap_up"
            reasoning = "노트 여부 재확인 (1회)"
            return {
                "next_step": next_step,
                "thinking_log": [ThinkingEntry(node="supervisor", reasoning=reasoning, decision=f"→ {next_step}")],
                "turn_count": turn_count,
                "note_requested": None,
                "wrap_up_retry_count": wrap_up_retry_count + 1,
            }

    # ── (C) Core routing (code-based) ──
    current_next_step = state.get("next_step")
    if turn_count == 0 and current_next_step == "verse_finder":
        next_step, reasoning = "verse_finder", "구절 없이 묵상 시작 — VerseFinder 노드"
    elif turn_count == 0:
        next_step, reasoning = "planner", "새 묵상 — 전략 수립"
    elif current_next_step == "verse_finder":
        next_step, reasoning = "verse_finder", "구절 탐색 계속"
    elif user_wants_end and depth < 90:
        next_step = "confirm_end"
        reasoning = f"종료 요청 but depth={depth} → 재확인"
    elif user_wants_end:
        next_step, reasoning = "wrap_up", f"종료 요청 depth={depth} → 마무리"
    elif depth >= 90:
        next_step, reasoning = "confirm_end", f"depth={depth} 충분 → 마무리 확인"
    elif turn_count >= 8:
        next_step, reasoning = "confirm_end", f"turn={turn_count} 충분 → 마무리 확인"
    else:
        next_step = "observer"
        reasoning = f"묵상 계속 (depth={depth}, turn={turn_count})"

    logger.info(f"[Supervisor] turn={turn_count} depth={depth} → {next_step}")

    return {
        "next_step": next_step,
        "thinking_log": [ThinkingEntry(node="supervisor", reasoning=reasoning, decision=f"→ {next_step}")],
        "turn_count": turn_count,
    }


# ──────────────────────────────────────────────
# 2. Planner Node
# ──────────────────────────────────────────────
async def planner_node(state: MeditationState) -> dict:
    """Analyzes the scripture passage and builds a question strategy using RAG.
    key_rag_insights를 압축 추출하여 Counselor에게 전달 (전체 RAG 대신).
    """
    scripture_ref = state.get("current_scripture", "")
    scripture_text = state.get("scripture_text", "")
    emotion = state.get("user_emotion", "")

    ref_book = ref_chapter = None
    try:
        parts = scripture_ref.split(":")
        if len(parts) >= 3:
            ref_book, ref_chapter = int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        pass

    rag_results = await search_theology(
        scripture_text,
        limit=5,
        ref_book=ref_book,
        ref_chapter=ref_chapter,
    )
    rag_context = format_rag_context(rag_results)

    llm = _get_llm(temperature=0.5)
    prompt = PLANNER_PROMPT.format(
        scripture_ref=scripture_ref,
        scripture_text=scripture_text,
        rag_context=rag_context or "(참조 자료 없음 — 본문 자체로 분석)",
        user_emotion=emotion or "미확인",
    )
    resp = await llm.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content="위 본문을 분석하고 질문 전략을 JSON으로 수립해주세요."),
    ])
    result = _parse_json_response(resp.content)

    strategy = result.get("question_strategy", [
        "이 본문에서 특별히 마음에 와닿는 단어나 구절이 있나요?",
        "그 말씀이 지금 당신의 삶에 어떤 의미로 다가오나요?",
        "이 말씀을 통해 오늘 어떤 기도를 드리고 싶으신가요?",
    ])
    # Planner가 압축한 핵심 인사이트 (100자 이내) — Counselor에 전달
    key_rag_insights = result.get("key_rag_insights", "") or ""

    thinking = ThinkingEntry(
        node="planner",
        reasoning=result.get("reasoning", "질문 전략 수립 완료"),
        decision=f"themes: {result.get('key_themes', [])}, questions: {len(strategy)}개",
    )
    logger.info(f"[Planner] strategy: {len(strategy)} questions, RAG hits: {len(rag_results)}")

    return {
        "question_strategy": strategy,
        "rag_context": rag_context,
        "key_rag_insights": key_rag_insights,
        "thinking_log": [thinking],
    }


# ──────────────────────────────────────────────
# 3. Counselor Node
# ──────────────────────────────────────────────
async def counselor_node(state: MeditationState) -> dict:
    """Converses with the user in Mate's warm tone.

    최적화:
    - 기존 MATE_SYSTEM_PROMPT(플레이스홀더 미사용 버그) + COUNSELOR_PROMPT(scripture 중복)
      → COUNSELOR_SYSTEM_PROMPT 단일 SystemMessage로 통합
    - RAG 전체 대신 planner가 압축한 key_rag_insights만 전달
    """
    scripture_ref = state.get("current_scripture", "")
    scripture_text = state.get("scripture_text", "")
    strategy = state.get("question_strategy", [])
    key_rag_insights = state.get("key_rag_insights", "") or "(신학 인사이트 없음)"
    turn_count = state.get("turn_count", 0)
    depth = state.get("meditation_depth", 0)
    emotion = state.get("user_emotion", "")

    system = COUNSELOR_SYSTEM_PROMPT.format(
        scripture_ref=scripture_ref,
        scripture_text=scripture_text,
        user_emotion=emotion or "미확인",
        question_strategy="\n".join(f"- {q}" for q in strategy) if strategy else "(자유 대화)",
        key_rag_insights=key_rag_insights,
        turn_count=turn_count,
        meditation_depth=depth,
    )

    llm = _get_llm(temperature=0.8)

    conversation_messages = list(state.get("messages", []))[-20:]
    if not conversation_messages or not any(isinstance(m, HumanMessage) for m in conversation_messages):
        conversation_messages.append(
            HumanMessage(content=f"묵상을 시작합니다. 본문: {scripture_ref}")
        )

    resp = await llm.ainvoke([
        SystemMessage(content=system),
        *conversation_messages,
    ])
    ai_response = _extract_text_content(resp.content)

    thinking = ThinkingEntry(
        node="counselor",
        reasoning=f"turn={turn_count}, 전략 기반 대화 생성",
        decision=f"응답: {ai_response[:50]}...",
    )
    logger.info(f"[Counselor] generated response ({len(ai_response)} chars)")

    return {
        "messages": [AIMessage(content=ai_response)],
        "turn_count": turn_count + 1,
        "thinking_log": [thinking],
        "last_executed_node": "counselor",
    }


# ──────────────────────────────────────────────
# 4. Observer Node
# ──────────────────────────────────────────────
async def observer_node(state: MeditationState) -> dict:
    """Measures the depth of the user's meditation response (0-100).
    간소화: 프롬프트 단축, HumanMessage 제거 (마지막 대화가 이미 컨텍스트에 포함됨).
    """
    current_depth = state.get("meditation_depth", 0)
    messages = state.get("messages", [])

    llm = _get_llm(temperature=0.2)
    prompt = OBSERVER_PROMPT.format(current_depth=current_depth)

    # 마지막 8개 메시지만 전달 (평가에 충분)
    resp = await llm.ainvoke([
        SystemMessage(content=prompt),
        *list(messages[-8:]),
    ])
    result = _parse_json_response(resp.content)

    new_depth = int(max(0, min(100, result.get("total_depth", current_depth))))
    suggestion = result.get("suggestion", "")

    thinking = ThinkingEntry(
        node="observer",
        reasoning=result.get("reasoning", result.get("assessment", "")),
        decision=f"depth: {current_depth} → {new_depth}, suggestion: {suggestion}",
    )
    logger.info(f"[Observer] depth: {current_depth} → {new_depth}")

    return {
        "meditation_depth": new_depth,
        "thinking_log": [thinking],
    }


# ──────────────────────────────────────────────
# 5. Scribe Node
# ──────────────────────────────────────────────
async def scribe_node(state: MeditationState) -> dict:
    """Compiles the conversation into a structured meditation note.
    간소화: SCRIBE_PROMPT에 scripture_text 제거 (대화 기록에 이미 포함됨).
    """
    scripture_ref = state.get("current_scripture", "")
    messages = state.get("messages", [])
    note_requested = state.get("note_requested", True)

    if note_requested is False:
        closing = "오늘 묵상을 함께해서 감사해요! 하나님의 말씀이 오늘 하루도 함께하시길 기도해요. 🙏 다음에 또 함께 묵상해요! 😊"
        thinking = ThinkingEntry(
            node="scribe",
            reasoning="사용자가 노트 생성 거절 — 간단한 마무리",
            decision="no note",
        )
        logger.info("[Scribe] user declined note")
        return {
            "messages": [AIMessage(content=closing)],
            "meditation_note": {"title": "오늘의 묵상", "skipped": True},
            "thinking_log": [thinking],
            "last_executed_node": "scribe",
        }

    llm = _get_llm(temperature=0.7)
    prompt = SCRIBE_PROMPT.format(scripture_ref=scripture_ref)

    try:
        resp = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="위 묵상 대화를 정리하여 JSON으로 묵상 노트를 작성해주세요. 반드시 1인칭('나')으로 작성하세요."),
            *list(messages[-20:]),
        ])
        result = _parse_json_response(resp.content)
    except Exception as e:
        logger.error(f"[Scribe] LLM call failed: {e}", exc_info=True)
        result = {}

    raw_text = result.get("raw", "")
    note = {
        "title": result.get("title", "오늘의 묵상"),
        "summary": result.get("summary", ""),
        "reflection": result.get("reflection", "") or raw_text[:300],
        "key_insights": result.get("key_insights", []),
        "prayer": result.get("prayer", ""),
        "closing_message": result.get("closing_message", "은혜로운 하루 되세요! 🙏"),
    }

    closing = f"📝 **{note['title']}**\n\n"
    closing += f"**한 줄 요약:** {note['summary']}\n\n"
    closing += f"**나의 묵상:** {note['reflection']}\n\n"
    if note["key_insights"]:
        closing += "**적용점:**\n" + "\n".join(f"• {i}" for i in note["key_insights"]) + "\n\n"
    closing += f"**기도문:**\n{note['prayer']}\n\n"
    closing += f"---\n{note['closing_message']}"

    thinking = ThinkingEntry(
        node="scribe",
        reasoning="묵상 완료 — 노트 작성",
        decision=f"title: {note['title']}",
    )
    logger.info(f"[Scribe] note: {note['title']}")

    return {
        "messages": [AIMessage(content=closing)],
        "meditation_note": note,
        "thinking_log": [thinking],
        "last_executed_node": "scribe",
    }


# ──────────────────────────────────────────────
# 6. Confirm End Node (LLM 제거 → 템플릿 기반)
# ──────────────────────────────────────────────
async def confirm_end_node(state: MeditationState) -> dict:
    """Asks the user to confirm ending meditation.

    최적화: LLM 호출 제거 → get_confirm_end_message() 템플릿 사용
    (세션당 LLM 1회 절약, 응답 속도 대폭 향상)
    """
    depth = state.get("meditation_depth", 0)
    turn_count = state.get("turn_count", 0)

    message = get_confirm_end_message(depth, turn_count)

    thinking = ThinkingEntry(
        node="confirm_end",
        reasoning=f"depth={depth}, 종료 의사 재확인 (템플릿)",
        decision="사용자에게 종료 여부 질문",
    )
    logger.info(f"[ConfirmEnd] depth={depth}, using template message")

    return {
        "messages": [AIMessage(content=message)],
        "turn_count": turn_count + 1,
        "thinking_log": [thinking],
        "last_executed_node": "confirm_end",
    }


# ──────────────────────────────────────────────
# 7. Wrap Up Node (LLM 제거 → 템플릿 기반)
# ──────────────────────────────────────────────
async def wrap_up_node(state: MeditationState) -> dict:
    """After meditation end is confirmed, asks about note creation.

    최적화: LLM 호출 제거 → WRAP_UP_MESSAGE 상수 사용
    (세션당 LLM 1회 절약, 응답 속도 대폭 향상)
    """
    turn_count = state.get("turn_count", 0)

    thinking = ThinkingEntry(
        node="wrap_up",
        reasoning="묵상 종료 확정 — 노트 여부 질문 (템플릿)",
        decision="노트 생성 여부 질문",
    )
    logger.info("[WrapUp] asking about note creation (template)")

    return {
        "messages": [AIMessage(content=WRAP_UP_MESSAGE)],
        "end_confirmed": True,
        "thinking_log": [thinking],
        "last_executed_node": "wrap_up",
    }
