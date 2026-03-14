"""
RAG 서비스 — 임베딩 생성 + 신학 벡터 검색 + 컨텍스트 조합
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings
from app.services.database import vector_search

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 임베딩 모델 (Singleton)
# ──────────────────────────────────────────────
_embeddings: GoogleGenerativeAIEmbeddings | None = None


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
        )
    return _embeddings


# ──────────────────────────────────────────────
# RAG 검색
# ──────────────────────────────────────────────
async def search_theology(
    query: str,
    *,
    limit: int = 5,
    ref_book: int | None = None,
    ref_chapter: int | None = None,
) -> list[dict[str, Any]]:
    """
    쿼리를 임베딩하고 theology_vectors에서 유사도 검색.
    반환: [{source_type, source_title, content_chunk, similarity, ...}]
    """
    try:
        embedding = await get_embeddings().aembed_query(query)
        results = await vector_search(
            embedding,
            limit=limit,
            ref_book=ref_book,
            ref_chapter=ref_chapter,
        )
        return results
    except Exception as e:
        logger.warning(f"RAG search failed (graceful degradation): {e}")
        return []


def format_rag_context(results: list[dict]) -> str:
    """
    RAG 검색 결과를 LLM 컨텍스트 문자열로 포맷팅.
    Base Layer (commentary) 60% + Spirit Layer (devotional) 40% 비율 적용.
    """
    if not results:
        return ""

    base_items = [r for r in results if r.get("source_type") in ("commentary", "dictionary", "confession")]
    spirit_items = [r for r in results if r.get("source_type") in ("sermon", "devotional", "prayer")]
    other_items = [r for r in results if r not in base_items and r not in spirit_items]

    # 비율 적용: base 60%, spirit 40%
    max_base = max(3, int(len(results) * 0.6))
    max_spirit = max(2, len(results) - max_base)

    selected = base_items[:max_base] + spirit_items[:max_spirit] + other_items
    selected = selected[: len(results)]  # 원래 limit 유지

    parts = []
    for i, r in enumerate(selected, 1):
        src = r.get("source_title") or r.get("source_type", "unknown")
        chunk = r.get("content_chunk", "")
        sim = r.get("similarity", 0)
        parts.append(f"[출처 {i}] ({src}, 유사도 {sim:.2f})\n{chunk}")

    return "\n\n".join(parts)


# ──────────────────────────────────────────────
# 구절 참조 검색 (cross-references)
# ──────────────────────────────────────────────
async def find_cross_references(
    verse_text: str,
    *,
    limit: int = 3,
) -> list[dict]:
    """본문 텍스트로 관련 구절 벡터 검색"""
    return await search_theology(verse_text, limit=limit)
