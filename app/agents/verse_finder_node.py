"""
VerseFinder Node — 구절 없이 묵상을 시작하는 사용자를 위한 노드

흐름:
1. 사용자 입력에서 구절 정보 파싱 시도
   - 파싱 성공 → verse_ref 추출 → 묵상 시작 (planner로 라우팅)
   - 파싱 실패 → 사용자 상황 파악 질문 또는 구절 추천

2. 구절 추천 흐름:
   - 사용자 상황/감정 파악
   - RAG 검색으로 어울리는 구절 찾기
   - 구절 3개 추천 + 간단한 이유 제시
   - 사용자가 선택하면 → 묵상 시작
"""

import json
import re
import logging
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.agents.nodes import _extract_text_content
from app.agents.state import MeditationState
from app.config import settings

logger = logging.getLogger(__name__)

# 한국어 성경책 이름과 숫자 인덱스 매핑 (KRV 기준)
KOREAN_BOOK_MAP = {
    "창세기": 1, "출애굽기": 2, "레위기": 3, "민수기": 4, "신명기": 5,
    "여호수아": 6, "사사기": 7, "룻기": 8, "사무엘상": 9, "사무엘하": 10,
    "열왕기상": 11, "열왕기하": 12, "역대상": 13, "역대하": 14, "에스라": 15,
    "느헤미야": 16, "에스더": 17, "욥기": 18, "시편": 19, "잠언": 20,
    "전도서": 21, "아가": 22, "이사야": 23, "예레미야": 24, "예레미야애가": 25,
    "에스겔": 26, "다니엘": 27, "호세아": 28, "요엘": 29, "아모스": 30,
    "오바댜": 31, "요나": 32, "미가": 33, "나훔": 34, "하박국": 35,
    "스바냐": 36, "학개": 37, "스가랴": 38, "말라기": 39,
    "마태복음": 40, "마가복음": 41, "누가복음": 42, "요한복음": 43, "사도행전": 44,
    "로마서": 45, "고린도전서": 46, "고린도후서": 47, "갈라디아서": 48,
    "에베소서": 49, "빌립보서": 50, "골로새서": 51, "데살로니가전서": 52,
    "데살로니가후서": 53, "디모데전서": 54, "디모데후서": 55, "디도서": 56,
    "빌레몬서": 57, "히브리서": 58, "야고보서": 59, "베드로전서": 60,
    "베드로후서": 61, "요한일서": 62, "요한이서": 63, "요한삼서": 64,
    "유다서": 65, "요한계시록": 66,
    # 약칭 매핑
    "창": 1, "출": 2, "레": 3, "민": 4, "신": 5, "수": 6, "삿": 7,
    "삼상": 9, "삼하": 10, "왕상": 11, "왕하": 12, "대상": 13, "대하": 14,
    "시": 19, "잠": 20, "전": 21, "사": 23, "렘": 24, "겔": 26, "단": 27,
    "마": 40, "막": 41, "눅": 42, "요": 43, "행": 44, "롬": 45,
    "고전": 46, "고후": 47, "갈": 48, "엡": 49, "빌": 50, "골": 51,
    "살전": 52, "살후": 53, "딤전": 54, "딤후": 55, "딛": 56, "히": 58,
    "약": 59, "벧전": 60, "벧후": 61, "요일": 62, "계": 66,
}


def parse_verse_ref_from_text(text: str) -> str | None:
    """
    사용자 입력 텍스트에서 성경 구절 참조를 파싱합니다.

    지원 형식:
    - "시편 23편 1절" → "KRV:19:23:1"
    - "요한복음 3:16" → "KRV:43:3:16"
    - "창세기 1장 1절" → "KRV:1:1:1"
    - "시 23:1" (약칭) → "KRV:19:23:1"

    반환: 파싱 성공 시 "KRV:{book}:{chapter}:{verse}" 형식, 실패 시 None
    """
    # 책 이름 패턴
    book_names = '|'.join(sorted(KOREAN_BOOK_MAP.keys(), key=len, reverse=True))

    # 패턴 1: "시편 23편 1절" 또는 "시편 23장 1절"
    pattern1 = rf'({book_names})\s*(\d+)[편장]\s*(\d+)절'
    match = re.search(pattern1, text)
    if match:
        book_name, chapter, verse = match.groups()
        book_num = KOREAN_BOOK_MAP.get(book_name)
        if book_num:
            return f"KRV:{book_num}:{chapter}:{verse}"

    # 패턴 2: "요한복음 3:16" 또는 "요 3:16"
    pattern2 = rf'({book_names})\s*(\d+):(\d+)'
    match = re.search(pattern2, text)
    if match:
        book_name, chapter, verse = match.groups()
        book_num = KOREAN_BOOK_MAP.get(book_name)
        if book_num:
            return f"KRV:{book_num}:{chapter}:{verse}"

    # 패턴 3: 장만 있는 경우 "시편 23편" → 1절로 시작
    pattern3 = rf'({book_names})\s*(\d+)[편장]'
    match = re.search(pattern3, text)
    if match:
        book_name, chapter = match.groups()
        book_num = KOREAN_BOOK_MAP.get(book_name)
        if book_num:
            return f"KRV:{book_num}:{chapter}:1"

    return None


async def verse_finder_node(state: MeditationState) -> dict:
    """
    구절 없이 묵상을 시작한 사용자를 위한 라우팅 노드.

    동작:
    1. 마지막 사용자 메시지에서 구절 파싱 시도
    2. 성공 → current_scripture 업데이트 + planner로 라우팅
    3. 실패 → LLM으로 구절 추천 또는 추가 질문
    """
    from app.config import settings

    messages = state.get("messages", [])
    if not messages:
        return {"next_step": "verse_finder"}  # 메시지 없으면 대기

    # 마지막 사용자 메시지 가져오기
    last_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    if not last_human:
        return {"next_step": "verse_finder"}

    # 1단계: 구절 파싱 시도
    verse_ref = parse_verse_ref_from_text(last_human)
    if verse_ref:
        # 구절을 찾았으면 DB에서 구절 텍스트도 가져온 뒤 묵상 시작
        logger.info(f"VerseFinder: 구절 파싱 성공 → {verse_ref}")
        from app.services.database import get_passage_text
        scripture_text = ""
        try:
            parts = verse_ref.split(":")
            if len(parts) >= 4:
                scripture_text = await get_passage_text(
                    parts[0], int(parts[1]), int(parts[2]), int(parts[3])
                ) or ""
        except Exception as e:
            logger.warning(f"VerseFinder: 구절 텍스트 로딩 실패 ({verse_ref}): {e}")

        return {
            "current_scripture": verse_ref,
            "scripture_text": scripture_text,  # ← Planner가 RAG/묵상 가이드에 사용
            "next_step": "planner",
            "messages": [AIMessage(content=f"'{last_human}'에서 구절을 찾았어요. 지금 바로 묵상을 시작할게요!")],
            "last_executed_node": "verse_finder",
        }

    # 2단계: 구절을 못 찾으면 LLM으로 상황 파악/추천
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.7,
        google_api_key=settings.gemini_api_key,
    )

    # 대화 히스토리 분석 (이전에 추천 요청이 있었는지 확인)
    conversation_turns = sum(1 for m in messages if isinstance(m, HumanMessage))

    if conversation_turns <= 2:
        # 첫 번째 또는 두 번째 대화 → 상황 파악 질문 또는 구절 추천
        prompt = f"""당신은 ABIDE 성경 묵상 도우미입니다.
사용자가 묵상할 구절을 아직 결정하지 못했습니다.
사용자 메시지: "{last_human}"

다음 중 하나를 수행하세요:
1. 메시지에서 감정이나 상황이 파악된다면 → 어울리는 성경 구절 2-3개 추천 (각 구절에 짧은 이유 포함)
2. 아직 파악이 안 된다면 → 현재 마음 상태나 고민을 한 가지만 물어보세요

추천 시 형식:
- 📖 [책명 장편 절절]: [구절 내용 일부]
  → [이 구절을 추천하는 이유 1-2문장]

마지막에 "어떤 구절로 묵상을 시작하고 싶으신가요?" 또는 선택을 유도하는 문장으로 마무리하세요.
한국어로 따뜻하게 답변해주세요."""
    else:
        # 여러 번 대화 후에도 구절 미결정 → 가장 적합한 구절 하나 강추
        prompt = f"""당신은 ABIDE 성경 묵상 도우미입니다.
사용자와 {conversation_turns}번 대화했지만 아직 구절을 결정하지 못했습니다.
사용자 마지막 메시지: "{last_human}"

지금까지 대화를 바탕으로 가장 적합한 구절 하나를 골라서 묵상을 시작하자고 제안하세요.
형식:
- 구절 제안 (책명 장:절)
- 이 구절을 선택한 이유 (1-2문장)
- "이 구절로 묵상을 시작할까요?" 질문

한국어로 따뜻하게 답변하세요."""

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return {
            "next_step": "verse_finder",  # 아직 구절 미결정, 계속 대기
            "messages": [AIMessage(content=_extract_text_content(response.content))],
            "last_executed_node": "verse_finder",
            "turn_count": state.get("turn_count", 0) + 1,
        }
    except Exception as e:
        # LLM 호출 실패 시 세션 크래시 방지 — 사용자에게 재시도 유도
        logger.error(f"VerseFinder LLM 호출 실패: {e}")
        return {
            "next_step": "verse_finder",
            "messages": [AIMessage(
                content="죄송합니다, 잠시 오류가 발생했어요. "
                        "묵상하고 싶은 구절을 다시 말씀해 주시겠어요?"
            )],
            "last_executed_node": "verse_finder",
            "turn_count": state.get("turn_count", 0) + 1,
        }
