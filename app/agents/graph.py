# Note: Some implementation details are not included in the public repository.
# See function docstrings for descriptions.
"""
Meditation Agent LangGraph Graph Definition

Graph flow:
  START → supervisor → {planner | observer | confirm_end | wrap_up | scribe}
  planner → counselor → END
  observer → counselor
  confirm_end → END  (awaits user response → supervisor decides in next turn)
  wrap_up → END      (awaits user response → supervisor routes to scribe)
  counselor → END
  scribe → END
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.agents.nodes import (
    _extract_text_content,
    confirm_end_node,
    counselor_node,
    observer_node,
    planner_node,
    scribe_node,
    supervisor_node,
    wrap_up_node,
)
from app.agents.verse_finder_node import verse_finder_node
from app.agents.state import MeditationState, create_initial_state
from app.services.database import get_passage_text

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Routing functions
# ──────────────────────────────────────────────
def route_from_verse_finder(state: MeditationState) -> Literal["planner", "verse_finder"]:
    """Route from verse_finder: planner when configured, verse_finder to await user"""
    return state.get("next_step", "verse_finder")

def route_from_supervisor(state: MeditationState) -> Literal["planner", "observer", "scribe", "confirm_end", "wrap_up", "verse_finder"]:
    """Route to next node based on Supervisor's decision"""
    return state.get("next_step", "planner")


# ──────────────────────────────────────────────
# Graph build
# ──────────────────────────────────────────────
def build_meditation_graph() -> StateGraph:
    """Build and compile the meditation agent LangGraph StateGraph."""

    graph = StateGraph(MeditationState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", planner_node)
    graph.add_node("counselor", counselor_node)
    graph.add_node("observer", observer_node)
    graph.add_node("scribe", scribe_node)
    graph.add_node("confirm_end", confirm_end_node)
    graph.add_node("wrap_up", wrap_up_node)
    graph.add_node("verse_finder", verse_finder_node)

    # Define edges
    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "planner": "planner",
            "observer": "observer",
            "scribe": "scribe",
            "confirm_end": "confirm_end",
            "wrap_up": "wrap_up",
            "verse_finder": "verse_finder",
        },
    )

    graph.add_edge("planner", "counselor")
    graph.add_edge("counselor", END)

    graph.add_edge("observer", "counselor")

    graph.add_edge("scribe", END)
    graph.add_edge("confirm_end", END)
    graph.add_edge("wrap_up", END)

    graph.add_conditional_edges(
        "verse_finder",
        lambda state: state.get("next_step", "verse_finder"),
        {
            "planner": "planner",
            "verse_finder": END,
        }
    )

    return graph.compile()


# Compiled graph (module-level singleton)
meditation_graph = build_meditation_graph()


# ──────────────────────────────────────────────
# Execution interface
# ──────────────────────────────────────────────
async def _fetch_single_verse_text(verse_ref: str) -> str:
    """
    Fetch text for a single verse reference.
    Supports ranges: "KRV:19:23:1-6" → Psalms 23:1–6
    """
    try:
        # Separate range: "KRV:19:23:1-6" → ref_part="KRV:19:23:1", end_verse=6
        end_verse = None
        ref_part = verse_ref
        
        # Check for range ("-") in last part
        parts = verse_ref.split(":")
        if parts and "-" in parts[-1]:
            verse_range = parts[-1].split("-")
            start_verse = int(verse_range[0])
            end_verse = int(verse_range[1])
            # Reconstruct verse_ref with start_verse only
            parts[-1] = str(start_verse)
            ref_part = ":".join(parts)
        
        parts = ref_part.split(":")
        if len(parts) >= 4:
            version, book, chapter, verse = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            ve = end_verse if end_verse else verse
            return await get_passage_text(version, book, chapter, verse, ve)
        elif len(parts) >= 3:
            version, book, chapter = parts[0], int(parts[1]), int(parts[2])
            return await get_passage_text(version, book, chapter, 1, 10)
    except Exception as e:
        logger.warning(f"Failed to fetch scripture text for {verse_ref}: {e}")
    return f"({verse_ref} — failed to load scripture text)"


async def start_meditation(
    *,
    user_id: str,
    session_id: str,
    verse_ref: str | None = None,
    verse_refs: list[str] | None = None,
    mood: str = "",
    session_type: str = "meditation",
    initial_query: str | None = None,
) -> dict[str, Any]:
    """
    Start a new meditation session.
    verse_ref format: "KRV:book:chapter:verse" (e.g. "KRV:19:23:1")
    """
    if session_type == "verse_finder" or (not verse_ref and initial_query):
        # VerseFinder mode
        initial_state = create_initial_state(
            user_id=user_id,
            session_id=session_id,
            verse_ref="",
            scripture_text="",
            mood=mood,
        )
        initial_state["next_step"] = "verse_finder"
        if initial_query:
            initial_state["messages"] = [HumanMessage(content=initial_query)]
        
        result = await meditation_graph.ainvoke(initial_state)
        return _extract_response(result, "")

    # Normal meditation mode
    # Collect multi-verse texts
    all_refs = verse_refs if verse_refs else ([verse_ref] if verse_ref else [])
    scripture_parts = []

    
    for ref in all_refs:
        text = await _fetch_single_verse_text(ref)
        scripture_parts.append(text)
    
    scripture_text = "\n\n".join(scripture_parts)
    
    # verse_ref used as main reference (for display)
    display_ref = verse_ref if not verse_refs else ", ".join(verse_refs)

    initial_state = create_initial_state(
        user_id=user_id,
        session_id=session_id,
        verse_ref=display_ref,
        scripture_text=scripture_text,
        mood=mood,
    )

    # Execute graph
    result = await meditation_graph.ainvoke(initial_state)

    return _extract_response(result, scripture_text)


async def continue_meditation(
    *,
    session_state: MeditationState,
    user_message: str,
) -> dict[str, Any]:
    """
    Add user message to an existing meditation session and run the agent.
    """
    # Add user message
    updated_state = dict(session_state)
    updated_state["messages"] = list(session_state.get("messages", [])) + [
        HumanMessage(content=user_message)
    ]
    # thinking_log은 _accumulate_list reducer로 자동 누적됨
    # 이번 턴의 로그만 수집하기 위해 초기화
    updated_state["thinking_log"] = []
    # last_executed_node는 이전 턴에서 설정된 값이 유지됨 (supervisor에서 참조)

    # Execute graph
    result = await meditation_graph.ainvoke(updated_state)

    return _extract_response(result, session_state.get("scripture_text", ""))


# 사용자에게 직접 보이는 응답을 LLM으로 생성하는 노드 — 이 노드의 LLM 토큰만 스트리밍
# wrap_up/confirm_end는 LLM 미사용(템플릿 AIMessage) → was_streamed=False로 별도 처리
_STREAMING_NODES = frozenset({"counselor", "scribe"})


async def start_meditation_streaming(
    *,
    user_id: str,
    session_id: str,
    verse_ref: str | None = None,
    verse_refs: list[str] | None = None,
    mood: str = "",
    session_type: str = "meditation",
    initial_query: str | None = None,
) -> AsyncGenerator[dict, None]:
    """
    start_meditation의 스트리밍 버전.
    {"type": "token", "content": "..."} — counselor/scribe의 LLM 토큰
    {"type": "result", "data": {...}}   — 그래프 완료 후 최종 결과 (was_streamed 플래그 포함)
    """
    if session_type == "verse_finder" or (not verse_ref and initial_query):
        initial_state = create_initial_state(
            user_id=user_id,
            session_id=session_id,
            verse_ref="",
            scripture_text="",
            mood=mood,
        )
        initial_state["next_step"] = "verse_finder"
        if initial_query:
            initial_state["messages"] = [HumanMessage(content=initial_query)]
        scripture_text = ""
    else:
        all_refs = verse_refs if verse_refs else ([verse_ref] if verse_ref else [])
        scripture_parts = []
        for ref in all_refs:
            text = await _fetch_single_verse_text(ref)
            scripture_parts.append(text)
        scripture_text = "\n\n".join(scripture_parts)
        display_ref = verse_ref if not verse_refs else ", ".join(verse_refs)
        initial_state = create_initial_state(
            user_id=user_id,
            session_id=session_id,
            verse_ref=display_ref,
            scripture_text=scripture_text,
            mood=mood,
        )

    final_output: MeditationState | None = None
    has_streamed = False

    async for event in meditation_graph.astream_events(initial_state, version="v2"):
        kind = event["event"]

        # counselor/scribe 노드의 LLM 토큰 → 즉시 스트리밍
        if kind == "on_chat_model_stream":
            node = event.get("metadata", {}).get("langgraph_node", "")
            if node in _STREAMING_NODES:
                chunk = event["data"]["chunk"]
                content = _extract_text_content(getattr(chunk, "content", ""))
                if content:
                    has_streamed = True
                    yield {"type": "token", "content": content}

        # 그래프 전체 완료 — 최종 상태 캡처
        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            final_output = event["data"].get("output")

    if final_output is not None:
        resp = _extract_response(final_output, scripture_text)
        resp["was_streamed"] = has_streamed
        yield {"type": "result", "data": resp}


async def continue_meditation_streaming(
    *,
    session_state: MeditationState,
    user_message: str,
) -> AsyncGenerator[dict, None]:
    """
    continue_meditation의 스트리밍 버전.
    """
    updated_state = dict(session_state)
    updated_state["messages"] = list(session_state.get("messages", [])) + [
        HumanMessage(content=user_message)
    ]
    updated_state["thinking_log"] = []

    scripture_text = session_state.get("scripture_text", "")
    final_output: MeditationState | None = None
    has_streamed = False

    async for event in meditation_graph.astream_events(updated_state, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            node = event.get("metadata", {}).get("langgraph_node", "")
            if node in _STREAMING_NODES:
                chunk = event["data"]["chunk"]
                content = _extract_text_content(getattr(chunk, "content", ""))
                if content:
                    has_streamed = True
                    yield {"type": "token", "content": content}

        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            final_output = event["data"].get("output")

    if final_output is not None:
        resp = _extract_response(final_output, scripture_text)
        resp["was_streamed"] = has_streamed
        yield {"type": "result", "data": resp}


def _extract_response(result: MeditationState, scripture_text: str) -> dict[str, Any]:
    """Extracts response data from the graph execution result."""
    messages = result.get("messages", [])

    ai_content = ""
    for m in reversed(messages):
        if hasattr(m, "content") and (not hasattr(m, "type") or m.type == "ai"):
            ai_content = m.content
            break

    return {
        "content": ai_content,
        "thinking_log": result.get("thinking_log", []),
        "meditation_depth": result.get("meditation_depth", 0),
        "turn_count": result.get("turn_count", 0),
        "meditation_note": result.get("meditation_note"),
        "referenced_verses": result.get("referenced_verses", []),
        "is_final": result.get("meditation_note") is not None,
        "state": dict(result),
        "scripture_text": scripture_text,
        "was_streamed": False,
    }
