"""
Meditation agent state schema (LangGraph State)
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ThinkingEntry(TypedDict):
    """Per-node chain-of-thought record"""
    node: str
    reasoning: str
    decision: str


class MeditationState(TypedDict):
    """Meditation agent shared state"""

    # Conversation messages (auto-accumulated via LangGraph add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Target scripture for meditation
    current_scripture: str       # verse_ref e.g. "KRV:19:23:1"
    scripture_text: str          # Scripture text

    # User context
    user_emotion: str            # Emotion state (joy, anxious, angry, tired, grateful, sad)
    user_id: str
    session_id: str

    # Meditation progress
    meditation_depth: int        # 0–100 depth score
    turn_count: int              # Conversation turn count
    next_step: Literal["planner", "observer", "counselor", "scribe", "confirm_end", "wrap_up"]

    # End-flow state
    end_confirmed: bool          # Whether the user confirmed they want to end
    note_requested: bool | None  # Whether the user wants a note generated (None = undecided)

    # Planner output
    question_strategy: list[str]  # Planned question points
    rag_context: str             # RAG search result context

    # Thinking log (for test client CoT monitoring)
    thinking_log: list[ThinkingEntry]

    # Scribe output (meditation note)
    meditation_note: dict[str, Any] | None

    # Referenced verses
    referenced_verses: list[dict]


# ──────────────────────────────────────────────
# Initial state creation helper
# ──────────────────────────────────────────────
def create_initial_state(
    *,
    user_id: str,
    session_id: str,
    verse_ref: str,
    scripture_text: str,
    mood: str = "",
) -> MeditationState:
    """Create initial state for a new meditation session."""
    return MeditationState(
        messages=[],
        current_scripture=verse_ref,
        scripture_text=scripture_text,
        user_emotion=mood,
        user_id=user_id,
        session_id=session_id,
        meditation_depth=0,
        turn_count=0,
        next_step="planner",
        end_confirmed=False,
        note_requested=None,
        question_strategy=[],
        rag_context="",
        thinking_log=[],
        meditation_note=None,
        referenced_verses=[],
    )
