# Note: Some implementation details are not included in the public repository.
# See function docstrings for descriptions.
"""
Meditation Agent Node Implementations (LangGraph Nodes)

7 nodes: Supervisor, Planner, Counselor, Observer, Scribe, ConfirmEnd, WrapUp
Each node receives MeditationState and returns a partial state update.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Note: Prompt templates imported from app/prompts/ (not included in public repo)
from app.prompts.prompts import (
    COUNSELOR_PROMPT,
    MATE_SYSTEM_PROMPT,
    OBSERVER_PROMPT,
    PLANNER_PROMPT,
    SCRIBE_PROMPT,
    SUPERVISOR_PROMPT,
    CONFIRM_END_PROMPT,
    WRAP_UP_PROMPT,
)
from app.agents.state import MeditationState, ThinkingEntry
from app.config import settings
from app.services.rag import format_rag_context, search_theology

logger = logging.getLogger(__name__)


def _get_llm(temperature: float = 0.7, json_mode: bool = False) -> ChatGoogleGenerativeAI:
    """Create an LLM instance"""
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "temperature": temperature,
        "google_api_key": settings.gemini_api_key,
        # streaming=True가 있어야 astream_events에서 on_chat_model_stream 토큰 이벤트가 발생함
        # json_mode는 구조화 출력이므로 스트리밍 불필요
        "streaming": not json_mode,
    }
    return ChatGoogleGenerativeAI(**kwargs)


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response (handles markdown code blocks)"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Strip first line (```json) and last line (```)
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
# 1. Supervisor Node (pure code-based routing)
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
    end_keywords = ["정리", "마무리", "기도문", "끝", "종료", "노트"]
    confirm_yes_keywords = ["네", "응", "끝낼게", "종료", "끝", "그래", "마무리", "좋아"]
    confirm_no_keywords = ["아니", "계속", "더", "아직", "이어서", "안 끝"]
    note_accept_keywords = ["노트", "만들어", "네", "좋아", "응", "부탁", "작성", "만들"]
    note_reject_keywords = ["아니", "괜찮", "안 만들", "됐어", "필요없", "다음에"]

    user_wants_end = any(kw in last_user_msg for kw in end_keywords)

    # Detect previous node from thinking_log
    last_thinking = state.get("thinking_log", [])
    was_confirm_end = any(t.get("node") == "confirm_end" for t in last_thinking) if last_thinking else False
    was_wrap_up = any(t.get("node") == "wrap_up" for t in last_thinking) if last_thinking else False

    # ── (A) After confirm_end: user responds yes/no ──
    if was_confirm_end and not end_confirmed:
        wants_end = any(kw in last_user_msg for kw in confirm_yes_keywords)
        wants_continue = any(kw in last_user_msg for kw in confirm_no_keywords)
        if wants_end:
            reasoning = "사용자가 묵상 종료를 확정했습니다. 마무리 절차로 이동합니다."
            next_step = "wrap_up"
            logger.info("[Supervisor] confirm_end → user confirmed end")
            return {
                "next_step": next_step,
                "thinking_log": [ThinkingEntry(node="supervisor", reasoning=reasoning, decision=f"next_step → {next_step}")],
                "turn_count": turn_count,
                "end_confirmed": True,
            }
        else:
            # User wants to continue (or ambiguous) → resume meditation
            reasoning = "사용자가 묵상을 계속하기로 했습니다." if wants_continue else "응답이 모호하지만 묵상을 계속합니다."
            next_step = "observer"
            logger.info("[Supervisor] confirm_end → user wants to continue")
            return {
                "next_step": next_step,
                "thinking_log": [ThinkingEntry(node="supervisor", reasoning=reasoning, decision=f"next_step → {next_step}")],
                "turn_count": turn_count,
                "end_confirmed": False,
            }

    # ── (B) After wrap_up: user responds about note ──
    if (was_wrap_up or end_confirmed) and note_requested is None and turn_count > 0:
        wants_note = any(kw in last_user_msg for kw in note_accept_keywords)
        rejects_note = any(kw in last_user_msg for kw in note_reject_keywords)
        if wants_note:
            reasoning = "사용자가 묵상 노트 생성을 요청했습니다."
            next_step = "scribe"
        elif rejects_note:
            reasoning = "사용자가 묵상 노트 없이 마무리를 원합니다."
            next_step = "scribe"
        else:
            reasoning = "노트 생성 여부를 다시 확인합니다."
            next_step = "wrap_up"

        logger.info(f"[Supervisor] end_confirmed, note decision → {next_step}")
        return {
            "next_step": next_step,
            "thinking_log": [ThinkingEntry(node="supervisor", reasoning=reasoning, decision=f"next_step → {next_step}")],
            "turn_count": turn_count,
            "note_requested": wants_note if (wants_note or rejects_note) else None,
        }

    # ── (C) Core routing logic (code-based, no LLM) ──
    if turn_count == 0:
        reasoning = "새 묵상 시작 — 전략 수립(planner)이 필요합니다."
        next_step = "planner"
    elif end_confirmed:
        reasoning = "묵상 종료 확정 — 노트 작성(scribe)으로 이동합니다."
        next_step = "scribe"
    elif user_wants_end and depth < 90:
        reasoning = f"사용자가 종료를 원하지만 깊이({depth}/100)가 아직 낮습니다. 종료 의사를 재확인합니다."
        next_step = "confirm_end"
    elif user_wants_end:
        reasoning = f"사용자가 종료를 원합니다 (depth={depth}) — 마무리로 이동합니다."
        next_step = "wrap_up"
    elif depth >= 90:
        reasoning = f"묵상 깊이가 충분합니다 (depth={depth}/100) — 마무리 여부를 확인합니다."
        next_step = "confirm_end"
    elif turn_count >= 8:
        reasoning = f"대화가 충분히 진행되었습니다 (turn={turn_count}) — 마무리 여부를 확인합니다."
        next_step = "confirm_end"
    else:
        reasoning = f"묵상 계속 진행 (depth={depth}, turn={turn_count})"
        next_step = "observer"

    thinking = ThinkingEntry(
        node="supervisor",
        reasoning=reasoning,
        decision=f"next_step → {next_step}",
    )

    logger.info(f"[Supervisor] turn={turn_count} depth={depth} → {next_step}")

    return {
        "next_step": next_step,
        "thinking_log": [thinking],
        "turn_count": turn_count,
    }


# ──────────────────────────────────────────────
# 2. Planner Node
# ──────────────────────────────────────────────
async def planner_node(state: MeditationState) -> dict:
    """Analyzes the scripture passage and builds a question strategy using RAG."""
    scripture_ref = state.get("current_scripture", "")
    scripture_text = state.get("scripture_text", "")
    emotion = state.get("user_emotion", "")

    ref_book = None
    ref_chapter = None
    try:
        parts = scripture_ref.split(":")
        if len(parts) >= 3:
            ref_book = int(parts[1])
            ref_chapter = int(parts[2])
    except (ValueError, IndexError):
        pass

    rag_results = await search_theology(
        scripture_text,
        limit=5,
        ref_book=ref_book,
        ref_chapter=ref_chapter,
    )
    rag_context = format_rag_context(rag_results)

    llm = _get_llm(temperature=0.5, json_mode=True)
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

    thinking = ThinkingEntry(
        node="planner",
        reasoning=result.get("reasoning", "질문 전략 수립 완료"),
        decision=f"themes: {result.get('key_themes', [])}, questions: {len(strategy)}개",
    )

    logger.info(f"[Planner] strategy: {len(strategy)} questions, RAG hits: {len(rag_results)}")

    return {
        "question_strategy": strategy,
        "rag_context": rag_context,
        "thinking_log": [thinking],
    }


# ──────────────────────────────────────────────
# 3. Counselor Node
# ──────────────────────────────────────────────
async def counselor_node(state: MeditationState) -> dict:
    """Persona node that converses with the user in Mate's warm, encouraging tone."""
    scripture_ref = state.get("current_scripture", "")
    scripture_text = state.get("scripture_text", "")
    strategy = state.get("question_strategy", [])
    rag_context = state.get("rag_context", "")
    turn_count = state.get("turn_count", 0)
    depth = state.get("meditation_depth", 0)
    emotion = state.get("user_emotion", "")

    system = MATE_SYSTEM_PROMPT.format(
        scripture_ref=scripture_ref,
        scripture_text=scripture_text,
        user_emotion=emotion or "미확인",
    )

    counselor_guide = COUNSELOR_PROMPT.format(
        scripture_ref=scripture_ref,
        scripture_text=scripture_text,
        question_strategy="\n".join(f"- {q}" for q in strategy) if strategy else "(자유 대화)",
        turn_count=turn_count,
        meditation_depth=depth,
        rag_context=rag_context or "(참조 자료 없음)",
    )

    llm = _get_llm(temperature=0.8)

    conversation_messages = list(state.get("messages", []))[-10:]

    if not conversation_messages or not any(isinstance(m, HumanMessage) for m in conversation_messages):
        conversation_messages.append(
            HumanMessage(content=f"묵상을 시작합니다. 본문: {scripture_ref}")
        )

    messages_for_llm = [
        SystemMessage(content=system),
        SystemMessage(content=counselor_guide),
        *conversation_messages,
    ]

    resp = await llm.ainvoke(messages_for_llm)
    ai_response = resp.content

    thinking = ThinkingEntry(
        node="counselor",
        reasoning=f"turn={turn_count}, strategy 기반 대화 생성",
        decision=f"응답 생성: {ai_response[:50]}...",
    )

    logger.info(f"[Counselor] generated response ({len(ai_response)} chars)")

    return {
        "messages": [AIMessage(content=ai_response)],
        "turn_count": turn_count + 1,
        "thinking_log": [thinking],
    }


# ──────────────────────────────────────────────
# 4. Observer Node
# ──────────────────────────────────────────────
async def observer_node(state: MeditationState) -> dict:
    """Measures the depth of the user's meditation response (0-100 score)."""
    current_depth = state.get("meditation_depth", 0)
    messages = state.get("messages", [])

    llm = _get_llm(temperature=0.3, json_mode=True)
    prompt = OBSERVER_PROMPT.format(current_depth=current_depth)

    resp = await llm.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content="위 본문과 묵상 내용을 분석하여 JSON으로 평가해주세요."),
    ] + list(messages[-8:]))
    result = _parse_json_response(resp.content)

    new_depth = result.get("total_depth", current_depth)
    new_depth = int(max(current_depth, min(100, new_depth)))

    assessment = result.get("assessment", "")
    suggestion = result.get("suggestion", "")

    thinking = ThinkingEntry(
        node="observer",
        reasoning=result.get("reasoning", assessment),
        decision=f"depth: {current_depth} → {new_depth}, suggestion: {suggestion}",
    )

    next_step = "counselor"

    logger.info(f"[Observer] depth: {current_depth} → {new_depth}, next → {next_step}")

    return {
        "meditation_depth": new_depth,
        "next_step": next_step,
        "thinking_log": [thinking],
    }


# ──────────────────────────────────────────────
# 5. Scribe Node
# ──────────────────────────────────────────────
async def scribe_node(state: MeditationState) -> dict:
    """Compiles the conversation into a structured meditation note with title, summary, insights, and prayer."""
    scripture_ref = state.get("current_scripture", "")
    scripture_text = state.get("scripture_text", "")
    messages = state.get("messages", [])
    note_requested = state.get("note_requested", True)

    if note_requested is False:
        closing = "오늘 묵상을 함께해서 감사해요! 하나님의 말씀이 오늘 하루도 함께하시길 기도해요. 🙏 다음에 또 함께 묵상해요! 😊"
        thinking = ThinkingEntry(
            node="scribe",
            reasoning="사용자가 노트 생성을 원하지 않아 간단한 마무리 메시지 생성",
            decision="no note, closing message only",
        )
        logger.info("[Scribe] user declined note, closing only")
        return {
            "messages": [AIMessage(content=closing)],
            "meditation_note": {"title": "오늘의 묵상", "skipped": True},
            "thinking_log": [thinking],
        }

    llm = _get_llm(temperature=0.7, json_mode=True)
    prompt = SCRIBE_PROMPT.format(
        scripture_ref=scripture_ref,
        scripture_text=scripture_text,
    )

    resp = await llm.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content="위 묵상 대화를 정리하여 JSON으로 묵상 노트를 작성해주세요."),
    ] + list(messages))
    result = _parse_json_response(resp.content)

    note = {
        "title": result.get("title", "오늘의 묵상"),
        "summary": result.get("summary", ""),
        "reflection": result.get("reflection", ""),
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

    logger.info(f"[Scribe] generated meditation note: {note['title']}")

    return {
        "messages": [AIMessage(content=closing)],
        "meditation_note": note,
        "thinking_log": [thinking],
    }


# ──────────────────────────────────────────────
# 6. Confirm End Node
# ──────────────────────────────────────────────
async def confirm_end_node(state: MeditationState) -> dict:
    """Asks the user to confirm ending the meditation when depth is still low."""
    depth = state.get("meditation_depth", 0)
    turn_count = state.get("turn_count", 0)
    messages = state.get("messages", [])

    llm = _get_llm(temperature=0.8)
    prompt = CONFIRM_END_PROMPT.format(
        meditation_depth=depth,
        turn_count=turn_count,
    )

    conversation_messages = list(messages[-6:])
    if not any(isinstance(m, HumanMessage) for m in conversation_messages):
        conversation_messages.append(HumanMessage(content="끝낼래"))

    resp = await llm.ainvoke([
        SystemMessage(content=prompt),
        *conversation_messages,
    ])

    thinking = ThinkingEntry(
        node="confirm_end",
        reasoning=f"depth={depth}, 종료 의사 재확인",
        decision="사용자에게 종료 여부 재문의",
    )

    logger.info(f"[ConfirmEnd] depth={depth}, asking user to confirm")

    return {
        "messages": [AIMessage(content=resp.content)],
        "turn_count": turn_count + 1,
        "thinking_log": [thinking],
    }


# ──────────────────────────────────────────────
# 7. Wrap Up Node
# ──────────────────────────────────────────────
async def wrap_up_node(state: MeditationState) -> dict:
    """After meditation end is confirmed, asks whether to create a meditation note or just wrap up."""
    depth = state.get("meditation_depth", 0)
    messages = state.get("messages", [])

    llm = _get_llm(temperature=0.8)
    prompt = WRAP_UP_PROMPT.format(
        meditation_depth=depth,
    )

    conversation_messages = list(messages[-4:])
    if not any(isinstance(m, HumanMessage) for m in conversation_messages):
        conversation_messages.append(HumanMessage(content="마무리할게"))

    resp = await llm.ainvoke([
        SystemMessage(content=prompt),
        *conversation_messages,
    ])

    thinking = ThinkingEntry(
        node="wrap_up",
        reasoning="묵상 종료 확정 — 노트 생성 여부 질문",
        decision="사용자에게 노트 생성 여부 질문",
    )

    logger.info("[WrapUp] asking user about note creation")

    return {
        "messages": [AIMessage(content=resp.content)],
        "end_confirmed": True,
        "thinking_log": [thinking],
    }
