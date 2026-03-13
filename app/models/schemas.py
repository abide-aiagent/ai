from pydantic import BaseModel, Field
from enum import Enum
from typing import Any


class SessionType(str, Enum):
    MEDITATION = "meditation"
    THEOLOGY_SEARCH = "theology_search"
    DEEP_LENS = "deep_lens"


# ──────────────────────────────────────────────
# Meditation (Mate) Schemas
# ──────────────────────────────────────────────
class MeditationStartRequest(BaseModel):
    """Meditation session start request"""
    user_id: str
    session_id: str
    verse_ref: str  # "KRV:19:23:1" (main verse)
    verse_refs: list[str] | None = None  # ["KRV:19:23:1-6", "KRV:43:3:16-18"] multi-verse
    mood: str | None = None


class MeditationChatRequest(BaseModel):
    """Meditation chat message"""
    session_id: str
    user_message: str


class MeditationResponse(BaseModel):
    """Meditation agent response"""
    session_id: str
    content: str
    thinking_log: list[dict] = Field(default_factory=list)
    meditation_depth: int = 0
    turn_count: int = 0
    referenced_verses: list[dict] = Field(default_factory=list)
    is_final: bool = False
    meditation_note: dict | None = None


# ──────────────────────────────────────────────
# Ask (Theology Search) Schemas
# ──────────────────────────────────────────────
class AskRequest(BaseModel):
    """Theology question (Ask) request"""
    user_id: str
    session_id: str
    query: str


class AskResponse(BaseModel):
    """Theology search response"""
    session_id: str
    content: str
    sources: list[dict] = Field(default_factory=list)
    related_verses: list[dict] = Field(default_factory=list)


# ──────────────────────────────────────────────
# DeepLens Schemas
# ──────────────────────────────────────────────
class DeepLensRequest(BaseModel):
    """DeepLens analysis request"""
    verse_ref: str  # "KRV:19:23:1"
    force_refresh: bool = False


class DeepLensResponse(BaseModel):
    """DeepLens analysis response"""
    verse_ref: str
    verse_text: str = ""
    context_guide: str = ""
    interpretation: str = ""
    application: str = ""
    cross_references: list[dict] = Field(default_factory=list)
    cached: bool = False


# ──────────────────────────────────────────────
# Compass Schemas
# ──────────────────────────────────────────────
class MoodCheckInRequest(BaseModel):
    """Mood check-in request"""
    user_id: str
    mood: str  # joy, anxious, angry, tired, grateful, sad


class MoodCheckInResponse(BaseModel):
    """Mood check-in response"""
    message: str
    recommended_verses: list[dict] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Report Classification (Admin Tool)
# ──────────────────────────────────────────────
class ReportClassifyRequest(BaseModel):
    """Report classification request"""
    report_reason: str
    target_content: str | None = None

class ReportClassifyResponse(BaseModel):
    """Report classification response"""
    severity: str  # low, medium, high, critical
    suggested_action: str
    confidence: float


# ──────────────────────────────────────────────
# TTS Generation (Meditation Guide)
# ──────────────────────────────────────────────
class TTSGenerateRequest(BaseModel):
    """TTS generation request"""
    text: str
    voice_type: str = "calm"

class TTSGenerateResponse(BaseModel):
    """TTS generation response"""
    audio_url: str
    duration: float


# ──────────────────────────────────────────────
# SSE Event Types
# ──────────────────────────────────────────────
class SSEEventType(str, Enum):
    MESSAGE = "message"
    VERSE_HIGHLIGHT = "verse_highlight"
    THINKING = "thinking"
    DONE = "done"
    ERROR = "error"
