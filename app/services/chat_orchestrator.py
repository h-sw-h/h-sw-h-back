# services/chat_orchestrator.py
from typing import List, Dict, Optional
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.services.chat_session import ChatSessionManager
from app.services.diary_service import DiaryService
from app.services.vector_store import VectorStoreService
from app.config import get_settings
import tiktoken

# ConversationSummaryBufferMemory 설정
MAX_TOKEN_LIMIT = 2000  # 최근 대화가 이 토큰 수를 초과하면 오래된 메시지 요약
SUMMARY_REDIS_KEY = "conversation_summary"  # Redis에 저장할 요약 키

class ChatOrchestrator:
    """
    채팅 플로우 오케스트레이션
    - 세션 관리
    - RAG 기반 일기 검색
    - 컨텍스트 구성
    - 모델 호출
    - 응답 저장
    """

    def __init__(
        self,
        session_manager: ChatSessionManager,
        diary_service: DiaryService,
        vector_store: Optional[VectorStoreService] = None
    ):
        self.session_manager = session_manager
        self.diary_service = diary_service
        self.vector_store = vector_store  # PDF 매뉴얼 RAG

        settings = get_settings()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7  # 공감적인 응답을 위해 조금 높게
        )
        # --- [주석] main_with_redis.py의 gpt-4o-mini 모델 설정을 가져옴 ---
        self.llm_mini = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.openai_api_key
        )
        # --- [주석] ---

        # 토큰 카운터 (gpt-4o-mini는 cl100k_base 인코딩 사용)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # CBT 기반 대화 시스템 프롬프트
        self.system_prompt = """당신은 은둔형 외톨이 청년의 사회복귀를 돕는 따뜻하고 공감적인 상담사입니다.

**역할:**
- 사용자의 감정을 경청하고 공감합니다
- 인지행동치료(CBT) 원리를 활용하여 대화합니다
- 작은 행동 변화를 격려합니다
- 판단하지 않고 있는 그대로 받아들입니다

**대화 가이드:**
1. **감정 탐색**: 사용자가 느끼는 감정을 먼저 파악하고 인정합니다
2. **생각 확인**: 어떤 생각이 그런 감정을 만들었는지 탐색합니다
3. **행동 제안**: 작고 실천 가능한 행동을 함께 찾아봅니다
4. **긍정 강화**: 작은 시도도 크게 격려합니다

**말투:**
- 존댓말 사용
- 짧고 명확한 문장
- 열린 질문 활용
- 따뜻하고 진솔한 톤

과거 대화 내역이나 유사한 일기가 있다면 참고하되, 현재 대화에 집중하세요."""

        # --- [주석] main_with_redis.py의 프롬프트 및 체인 설정 ---
        # 1. 프롬프트 템플릿 정의
        cbt_extract_prompt_template = """
        당신은 CBT(인지행동치료) 전문가입니다.
        [대화 전사]를 읽고, [상황, 생각, 감정, 행동] 4가지 요소를 JSON으로 추출하세요.
        [규칙]...
        [대화 전사]
        {transcript}
        [출력 형식 (JSON)]
        {{
          "situation": "...",
          "thoughts": [...],
          "emotions": [...],
          "behaviors": [...]
        }}
        """
        self.cbt_extract_prompt = ChatPromptTemplate.from_template(cbt_extract_prompt_template)

        alt_perspective_prompt_template = """
        당신은 친절한 CBT 코치입니다.
        [자동적 사고]를 완화할 '다른 관점'을 1~2문장의 조언으로 작성해 주세요.
        [자동적 사고]
        {thoughts_text}
        [생성할 '다른 관점']
        """
        self.alt_perspective_prompt = ChatPromptTemplate.from_template(alt_perspective_prompt_template)

        diary_generation_prompt_template = """
        당신은 '일기 작성가'입니다.
        주어진 [CBT 분석 데이터 (S-T-E-B)]를 바탕으로, 1인칭 '간단한 하루 일기'를 작성해 주세요.
        조언은 포함하지 말고, 오직 사용자의 경험(S-T-E-B)만 서술하세요.
        [CBT 분석 데이터]
        {cbt_json_data}
        [작성할 일기]
        """
        self.diary_generation_prompt = ChatPromptTemplate.from_template(diary_generation_prompt_template)

        # 2. 파서 및 LangChain 체인 구성
        string_parser = StrOutputParser()

        self.chain_extract_cbt = self.cbt_extract_prompt | self.llm_mini | string_parser
        self.chain_gen_perspective = self.alt_perspective_prompt | self.llm_mini | string_parser
        self.chain_create_diary = self.diary_generation_prompt | self.llm_mini | string_parser
        # --- [주석] ---

    # ------------------------
    # 외부에서 호출하는 메인 엔드포인트
    # ------------------------
    def process_message(
        self,
        session_id: str,
        user_message: str
    ) -> Dict:
        """
        사용자 메시지 처리 (전체 플로우)

        **Redis + ConversationSummaryBufferMemory 통합:**
        - Redis: 전체 대화 영속화 스토리지
        - ConversationSummaryBufferMemory 로직: 오래된 메시지 자동 요약, 최근 메시지 원본 유지

        Args:
            session_id: 세션 ID
            user_message: 사용자 메시지

        Returns:
            응답 데이터 (answer, sources)
        """
        # 1. 세션 존재 확인
        if not self.session_manager.session_exists(session_id):
            raise ValueError("유효하지 않은 세션입니다")

        # 2. 세션 정보 조회 (user_id 가져오기)
        session_info = self.session_manager.get_session_info(session_id)
        user_id = session_info.get("user_id")

        # 3. Redis에서 기존 대화 내역 전체 로드
        full_conversation = self.session_manager.get_full_conversation(session_id)

        # 4. ConversationSummaryBufferMemory 로직 적용 (수동 구현)
        buffered_messages = self._apply_summary_buffer_memory(session_id, full_conversation)

        # 5. 과거 일기 검색 (RAG)
        similar_diaries = self.diary_service.search_similar_diaries(
            user_id=user_id, query=user_message, k=3
        )

        # 6. PDF 매뉴얼 검색 (RAG)
        manual_context = None
        if self.vector_store:
            try:
                manual_result = self.vector_store.query(user_message)
                manual_context = manual_result.get("answer", "")
            except Exception as e:
                print(f"매뉴얼 검색 실패: {e}")

        # 7. 컨텍스트 구성 (시스템 프롬프트 + RAG + 버퍼된 대화)
        context = self._build_context_with_memory(
            similar_diaries=similar_diaries,
            buffered_messages=buffered_messages,
            current_message=user_message,
            manual_context=manual_context
        )

        # 8. 모델 호출
        assistant_response = self._generate_response(context)

        # 9. Redis에 저장 (영속화)
        self.session_manager.add_message(session_id, "user", user_message)
        self.session_manager.add_message(session_id, "assistant", assistant_response)

        return {
            "answer": assistant_response,
            "similar_diaries": [d["metadata"].get("created_at") for d in similar_diaries] if similar_diaries else None
        }

    def _apply_summary_buffer_memory(
        self,
        session_id: str,
        full_conversation: List[Dict]
    ) -> List:
        """
        ConversationSummaryBufferMemory 로직 적용

        **동작 원리:**
        1. 최근 메시지들의 토큰 수 계산
        2. MAX_TOKEN_LIMIT 초과 시:
           - 오래된 메시지들을 LLM으로 요약
           - 요약을 Redis에 캐시 (중복 요약 방지)
           - 요약 + 최근 원본 메시지 반환
        3. 미만이면 전체 원본 메시지 반환

        Returns:
            Message 객체 리스트 (SystemMessage(요약) + 최근 HumanMessage/AIMessage)
        """
        if not full_conversation:
            return []

        # 1. 최근 메시지부터 역순으로 토큰 누적 계산
        recent_messages = []
        recent_token_count = 0

        for msg in reversed(full_conversation):
            msg_tokens = len(self.tokenizer.encode(msg["content"]))

            if recent_token_count + msg_tokens <= MAX_TOKEN_LIMIT:
                recent_messages.insert(0, msg)  # 앞에 삽입 (원래 순서 유지)
                recent_token_count += msg_tokens
            else:
                break  # 토큰 한계 초과

        # 2. 요약이 필요한지 확인
        old_messages = full_conversation[:len(full_conversation) - len(recent_messages)]

        if not old_messages:
            # 요약 불필요 - 최근 메시지만 반환
            return self._convert_to_langchain_messages(recent_messages)

        # 3. Redis에서 기존 요약 확인
        summary_key = f"session:{session_id}"
        cached_summary = self.session_manager.redis.hget(summary_key, SUMMARY_REDIS_KEY)

        # 4. 요약이 없거나 오래된 메시지가 추가되었으면 새로 요약
        cached_msg_count = self.session_manager.redis.hget(summary_key, "summarized_count")

        if not cached_summary or (cached_msg_count and int(cached_msg_count) < len(old_messages)):
            print(f"[SummaryBuffer] 오래된 메시지 {len(old_messages)}개 요약 중...")

            # LLM으로 오래된 메시지 요약
            summary_text = self._summarize_old_messages(old_messages)

            # Redis에 캐시
            self.session_manager.redis.hset(summary_key, SUMMARY_REDIS_KEY, summary_text)
            self.session_manager.redis.hset(summary_key, "summarized_count", len(old_messages))

            print(f"[SummaryBuffer] 요약 완료 및 Redis 캐시 저장")
        else:
            summary_text = cached_summary.decode('utf-8') if isinstance(cached_summary, bytes) else cached_summary
            print(f"[SummaryBuffer] Redis 캐시에서 요약 로드 (메시지 {len(old_messages)}개)")

        # 5. 요약 메시지 + 최근 원본 메시지 반환
        buffered_messages = [SystemMessage(content=f"**이전 대화 요약:**\n{summary_text}")]
        buffered_messages.extend(self._convert_to_langchain_messages(recent_messages))

        return buffered_messages

    def _convert_to_langchain_messages(self, messages: List[Dict]) -> List:
        """Redis 메시지를 LangChain Message 객체로 변환"""
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        return langchain_messages

    def _summarize_old_messages(self, old_messages: List[Dict]) -> str:
        """
        오래된 메시지들을 LLM으로 요약

        Args:
            old_messages: 요약할 메시지 리스트

        Returns:
            요약 텍스트
        """
        # 대화 텍스트 구성
        conversation_text = ""
        for msg in old_messages:
            role = "사용자" if msg["role"] == "user" else "상담사"
            conversation_text += f"{role}: {msg['content']}\n\n"

        # 요약 프롬프트
        summary_prompt = f"""다음은 상담 대화의 초기 부분입니다. 이를 간결하게 요약해주세요.

**대화 내용:**
{conversation_text}

**요약 지침:**
- 핵심 주제와 감정만 포함
- 3-5 문장으로 간결하게
- 사용자의 관점에서 작성

요약:"""

        messages = [
            SystemMessage(content="당신은 상담 대화를 요약하는 전문가입니다."),
            HumanMessage(content=summary_prompt)
        ]

        response = self.llm.invoke(messages)
        return response.content.strip()

    def _build_context_with_memory(
        self,
        similar_diaries: List[Dict],
        buffered_messages: List,
        current_message: str,
        manual_context: Optional[str] = None
    ) -> List:
        """
        컨텍스트 구성 (시스템 프롬프트 + PDF 매뉴얼 + 과거 일기 + ConversationSummaryBufferMemory + 현재 메시지)

        Args:
            similar_diaries: RAG로 검색된 유사 일기
            buffered_messages: ConversationSummaryBufferMemory에서 가져온 메시지 (요약 + 최근 원본)
            current_message: 현재 사용자 메시지
            manual_context: PDF 매뉴얼 컨텍스트
        """
        messages = []

        # 1. 시스템 프롬프트
        system_content = self.system_prompt

        # 2. PDF 매뉴얼 전문 지식 추가 (있으면)
        if manual_context:
            knowledge_context = f"\n\n**전문 지식 (참고 자료):**\n{manual_context}\n"
            system_content += "\n" + knowledge_context

        # 3. 유사 일기 추가 (있으면)
        if similar_diaries:
            diary_context = "\n\n**과거 일기 참고:**\n"
            for idx, diary in enumerate(similar_diaries, 1):
                created_at = diary["metadata"].get("created_at", "알 수 없음")
                content = diary["content"][:200]  # 처음 200자만
                diary_context += f"{idx}. [{created_at}] {content}...\n"

            system_content += "\n" + diary_context

        messages.append(SystemMessage(content=system_content))

        # 4. ConversationSummaryBufferMemory에서 가져온 버퍼된 대화 내역 추가
        # (자동으로 요약된 과거 대화 + 최근 원본 메시지)
        messages.extend(buffered_messages)

        # 5. 현재 메시지
        messages.append(HumanMessage(content=current_message))

        return messages

    def _generate_response(self, messages: List) -> str:
        """
        LLM을 호출하여 응답 생성
        """
        response = self.llm.invoke(messages)
        return response.content

    # --- [주석] main_with_redis.py 로직을 적용하여 수정한 일기 생성 메서드 ---
    def _extract_json_from_markdown(self, text: str) -> Optional[str]:
        """
        AI가 반환한 마크다운(```json ... ```) 텍스트에서
        순수한 JSON 문자열({ ... })만 추출합니다.
        """
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        else:
            if text.strip().startswith("{"):
                return text
        return None

    def summarize_conversation_to_diary(self, session_id: str) -> Dict[str, str]:
        """
        대화 요약 → CBT 4요소 추출 → 일기 및 다른 관점 생성

        Returns:
            생성된 일기 및 다른 관점을 포함한 딕셔너리
        """
        # 1. 전체 대화 내용 가져오기
        full_conversation = self.session_manager.get_full_conversation(session_id)
        if not full_conversation:
            return {
                "diary_text": "일기를 생성할 대화 내용이 없습니다.",
                "alternative_perspective": ""
            }

        # 대화 내용을 하나의 문자열로 변환
        transcript = "\n".join([f"{'사용자' if msg['role'] == 'user' else '상담사'}: {msg['content']}" for msg in full_conversation])

        try:
            # 2. LLM을 통해 대화 내용에서 CBT 4요소(S-T-E-B) 추출
            cbt_data_str = self.chain_extract_cbt.invoke({
                "transcript": transcript
            })

            # 3. AI가 생성한 응답에서 순수 JSON 부분만 추출
            pure_json_str = self._extract_json_from_markdown(cbt_data_str)
            if not pure_json_str:
                error_message = f"오류: AI 응답에서 CBT 데이터를 추출하지 못했습니다. (응답: {cbt_data_str})"
                print(error_message)
                return {
                    "diary_text": "일기 생성 중 오류가 발생했습니다. 대화 내용을 분석하는 데 실패했습니다.",
                    "alternative_perspective": error_message
                }

            # 4. 추출된 JSON 문자열을 파이썬 딕셔너리로 변환
            try:
                cbt_data = json.loads(pure_json_str)
            except json.JSONDecodeError:
                error_message = f"오류: AI가 생성한 CBT 데이터의 형식이 잘못되었습니다. (내용: {pure_json_str})"
                print(error_message)
                return {
                    "diary_text": "일기 생성 중 오류가 발생했습니다. 분석된 데이터 형식이 올바르지 않습니다.",
                    "alternative_perspective": error_message
                }

            # 5. 추출된 '자동적 사고' 목록을 바탕으로 '다른 관점' 생성
            thoughts_list = cbt_data.get('thoughts', [])
            thought_texts = []
            for t in thoughts_list:
                if isinstance(t, dict):
                    thought_texts.append(t.get('text', ''))
                elif isinstance(t, str):
                    thought_texts.append(t)
            
            final_alternative_perspective = ""
            if thought_texts:
                final_alternative_perspective = self.chain_gen_perspective.invoke({
                    "thoughts_text": "\n- ".join(thought_texts)
                })

            # 6. 추출된 CBT 데이터를 바탕으로 1인칭 시점의 일기 생성
            final_diary_text = self.chain_create_diary.invoke({
                "cbt_json_data": json.dumps(cbt_data, ensure_ascii=False)
            })

            # 7. 최종 결과 반환
            return {
                "diary_text": final_diary_text,
                "alternative_perspective": final_alternative_perspective
            }

        except Exception as e:
            error_message = f"일기 생성 중 예기치 않은 오류 발생: {str(e)}"
            print(error_message)
            return {
                "diary_text": "일기 생성 중 알 수 없는 오류가 발생했습니다.",
                "alternative_perspective": error_message
            }
    # --- [주석] ---


# 전역 오케스트레이터 인스턴스
_orchestrator: Optional[ChatOrchestrator] = None

def get_chat_orchestrator(
    session_manager: Optional[ChatSessionManager] = None,
    diary_service: Optional[DiaryService] = None,
    vector_store: Optional[VectorStoreService] = None
) -> ChatOrchestrator:
    """
    채팅 오케스트레이터 의존성 주입
    """
    global _orchestrator

    if _orchestrator is None:
        from app.services.chat_session import get_session_manager
        from app.services.diary_service import get_diary_service

        sm = session_manager or get_session_manager()
        ds = diary_service or get_diary_service()

        # vector_store는 chatbot 라우터에서 초기화된 전역 인스턴스 사용
        _orchestrator = ChatOrchestrator(sm, ds, vector_store)

    return _orchestrator
