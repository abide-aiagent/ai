# Note: Some implementation details are not included in the public repository.
# See function docstrings for descriptions.
"""
Meditation Agent LangGraph Graph Definition

Graph flow:
  START → supervisor → {planner | observer | confirm_end | wrap_up | scribe}
  planner → counselor → END
  observer → {counselor | wrap_up}
  confirm_end → END  (awaits user response → supervisor decides in next turn)
  wrap_up → END      (awaits user response → supervisor routes to scribe)
  counselor → END
  scribe → END
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.agents.nodes import (
    confirm_end_node,
    counselor_node,
    observer_node,
    planner_node,
    scribe_node,
    supervisor_node,
    wrap_up_node,
)
from app.agents.state import MeditationState, create_initial_state
from app.services.database import get_passage_text

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Routing functions
# ──────────────────────────────────────────────
def route_from_supervisor(state: MeditationState) -> Literal["planner", "observer", "scribe", "confirm_end", "wrap_up"]:
    """Route to next node based on Supervisor's decision"""
    return state.get("next_step", "planner")


def route_from_observer(state: MeditationState) -> Literal["counselor", "wrap_up"]:
    """Continue conversation or wrap up based on Observer's evaluation"""
    return state.get("next_step", "counselor")


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
        },
    )

    graph.add_edge("planner", "counselor")
    graph.add_edge("counselor", END)

    graph.add_conditional_edges(
        "observer",
        route_from_observer,
        {
            "counselor": "counselor",
            "wrap_up": "wrap_up",
        },
    )

    graph.add_edge("scribe", END)
    graph.add_edge("confirm_end", END)
    graph.add_edge("wrap_up", END)

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
    verse_ref: str,
    verse_refs: list[str] | None = None,
    mood: str = "",
) -> dict[str, Any]:
    """
    Start a new meditation session.
    verse_ref format: "KRV:book:chapter:verse" (e.g. "KRV:19:23:1")
    verse_refs: multi-verse support (e.g. ["KRV:19:23:1-6", "KRV:43:3:16-18"])
    """
    # Collect multi-verse texts
    all_refs = verse_refs if verse_refs else [verse_ref]
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
    # Reset thinking_log (only collect logs for this turn)
    updated_state["thinking_log"] = []

    # Execute graph
    result = await meditation_graph.ainvoke(updated_state)

    return _extract_response(result, session_state.get("scripture_text", ""))


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
    }
