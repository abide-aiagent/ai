"""
Database service — async PostgreSQL connection pool and helper functions
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import psycopg
import psycopg.rows
from psycopg_pool import AsyncConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Connection pool (initialised/closed in app lifespan)
# ──────────────────────────────────────────────
_pool: AsyncConnectionPool | None = None


async def init_pool() -> None:
    """Called at app startup — create connection pool"""
    global _pool
    _pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=2,
        max_size=10,
        open=False,
    )
    await _pool.open()
    logger.info("Database pool opened")


async def close_pool() -> None:
    """Called at app shutdown — close connection pool"""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Database pool closed")


def get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("Database pool not initialized — call init_pool() first")
    return _pool


@asynccontextmanager
async def get_conn():
    """Convenience connection context manager"""
    async with get_pool().connection() as conn:
        yield conn


# ──────────────────────────────────────────────
# Bible query helpers
# ──────────────────────────────────────────────
async def get_verse(version: str, book: int, chapter: int, verse: int) -> dict | None:
    """Fetch a single verse"""
    async with get_conn() as conn:
        row = await conn.execute(
            "SELECT content FROM bibles WHERE version=%s AND book=%s AND chapter=%s AND verse=%s",
            (version, book, chapter, verse),
        )
        result = await row.fetchone()
        return {"content": result[0]} if result else None


async def get_chapter(version: str, book: int, chapter: int) -> list[dict]:
    """Fetch all verses in a chapter"""
    async with get_conn() as conn:
        cur = await conn.execute(
            "SELECT verse, content FROM bibles WHERE version=%s AND book=%s AND chapter=%s ORDER BY verse",
            (version, book, chapter),
        )
        rows = await cur.fetchall()
        return [{"verse": r[0], "content": r[1]} for r in rows]


async def get_passage_text(version: str, book: int, chapter: int, verse_start: int, verse_end: int | None = None) -> str:
    """Return a range of verses as a single string"""
    if verse_end is None:
        verse_end = verse_start
    async with get_conn() as conn:
        cur = await conn.execute(
            "SELECT verse, content FROM bibles WHERE version=%s AND book=%s AND chapter=%s AND verse BETWEEN %s AND %s ORDER BY verse",
            (version, book, chapter, verse_start, verse_end),
        )
        rows = await cur.fetchall()
        return " ".join(f"{r[0]}절 {r[1]}" for r in rows)


# ──────────────────────────────────────────────
# Vector search (RAG)
# ──────────────────────────────────────────────
async def vector_search(
    embedding: list[float],
    *,
    limit: int = 5,
    ref_book: int | None = None,
    ref_chapter: int | None = None,
) -> list[dict[str, Any]]:
    """
    Cosine similarity search on theology_vectors table.
    Uses pgvector's <=> operator (cosine distance).
    """
    params: list[Any] = [str(embedding)]
    where_clauses = []

    if ref_book is not None:
        where_clauses.append("ref_book = %s")
        params.append(ref_book)
    if ref_chapter is not None:
        where_clauses.append("ref_chapter = %s")
        params.append(ref_chapter)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    params.append(limit)

    query = f"""
        SELECT id, source_type, source_title, ref_book, ref_chapter, ref_verse,
               content_chunk, metadata,
               1 - (embedding <=> %s::vector) AS similarity
        FROM theology_vectors
        {where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    # Build explicit parameter list (embedding used twice: in SELECT and ORDER BY)
    embedding_str = str(embedding)
    full_params: list[Any] = [embedding_str]  # For similarity calculation in SELECT
    if ref_book is not None:
        full_params.append(ref_book)
    if ref_chapter is not None:
        full_params.append(ref_chapter)
    full_params.append(embedding_str)  # For ORDER BY sorting
    full_params.append(limit)

    async with get_conn() as conn:
        cur = await conn.execute(query, full_params)
        rows = await cur.fetchall()
        cols = [d.name for d in cur.description] if cur.description else []
        return [dict(zip(cols, row)) for row in rows]


# ──────────────────────────────────────────────
# DeepLens cache
# ──────────────────────────────────────────────
async def get_deep_lens_cache(verse_ref: str) -> dict | None:
    """Fetch cached DeepLens analysis"""
    async with get_conn() as conn:
        cur = await conn.execute(
            """
            SELECT context_guide, interpretation, application, cross_references, hit_count
            FROM deep_lens_cache
            WHERE verse_ref = %s AND expires_at > NOW()
            """,
            (verse_ref,),
        )
        row = await cur.fetchone()
        if row:
            # Increment hit_count
            await conn.execute(
                "UPDATE deep_lens_cache SET hit_count = hit_count + 1 WHERE verse_ref = %s",
                (verse_ref,),
            )
            return {
                "context_guide": row[0],
                "interpretation": row[1],
                "application": row[2],
                "cross_references": row[3],
                "hit_count": row[4] + 1,
            }
        return None


async def save_deep_lens_cache(
    verse_ref: str,
    context_guide: str,
    interpretation: str,
    application: str,
    cross_references: list[dict] | None = None,
) -> None:
    """Save DeepLens analysis result to cache"""
    import json

    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO deep_lens_cache (verse_ref, context_guide, interpretation, application, cross_references)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (verse_ref) DO UPDATE SET
                context_guide = EXCLUDED.context_guide,
                interpretation = EXCLUDED.interpretation,
                application = EXCLUDED.application,
                cross_references = EXCLUDED.cross_references,
                hit_count = 0,
                created_at = NOW(),
                expires_at = NOW() + INTERVAL '30 days'
            """,
            (
                verse_ref,
                context_guide,
                interpretation,
                application,
                json.dumps(cross_references or [], ensure_ascii=False),
            ),
        )


# ──────────────────────────────────────────────
# Session & Messages
# ──────────────────────────────────────────────
async def ensure_session_exists(
    session_id: str,
    user_id: str | None = None,
    session_type: str = "meditation",
    verse_ref: str = "",
) -> None:
    """
    Ensure ai_sessions row exists (upsert).
    Needed when AI server is called directly without Core server (e.g. test client).
    user_id is set NULL to avoid users table FK dependency.
    """
    try:
        async with get_conn() as conn:
            await conn.execute(
                """
                INSERT INTO ai_sessions (id, session_type, target_verse_ref)
                VALUES (%s::uuid, %s::session_type, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (session_id, session_type, verse_ref),
            )
    except Exception as e:
        logger.warning(f"ensure_session_exists failed (non-critical): {e}")


async def save_ai_message(
    session_id: str,
    role: str,
    content: str,
    referenced_verses: list[dict] | None = None,
    tokens_used: int = 0,
    latency_ms: int | None = None,
) -> int:
    """Save AI message to DB and return its id"""
    import json

    try:
        async with get_conn() as conn:
            cur = await conn.execute(
                """
                INSERT INTO ai_messages (session_id, role, content, referenced_verses, tokens_used, latency_ms)
                VALUES (%s::uuid, %s, %s, %s::jsonb, %s, %s)
                RETURNING id
                """,
                (
                    session_id,
                    role,
                    content,
                    json.dumps(referenced_verses or [], ensure_ascii=False),
                    tokens_used,
                    latency_ms,
                ),
            )
            row = await cur.fetchone()
            return row[0] if row else 0
    except Exception as e:
        logger.warning(f"save_ai_message failed (session_id={session_id}): {e}")
        return 0


# ──────────────────────────────────────────────
# Mood Verses (from admin-managed mood_verses table)
# ──────────────────────────────────────────────
async def get_mood_verses_from_db(mood: str) -> list[dict]:
    """Fetch mood-based verse recommendations from the DB."""
    try:
        async with get_conn() as conn:
            cur = await conn.execute(
                """
                SELECT verse_ref, verse_text, tag
                FROM mood_verses
                WHERE mood = %s AND is_active = TRUE
                ORDER BY display_order
                """,
                (mood,),
            )
            rows = await cur.fetchall()
            if rows:
                return [{"ref": r[0], "text": r[1], "tag": r[2]} for r in rows]
    except Exception as e:
        logger.warning(f"get_mood_verses_from_db failed (mood={mood}): {e}")
    return []
