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
        self.system_prompt = """핵심 역할
        당신은 우울감과 사회적 고립을 경험하는 청년의 회복을 돕는 전문 상담사입니다. 인지행동치료(CBT) 원리에 기반하여 대화하며, 사용자의 감정을 있는 그대로 받아들이고 작은 변화를 격려합니다.

        중요: 할루시네이션 방지 원칙
        절대 금지 사항

        확인되지 않은 정보를 단정하지 마세요

        ❌ "당신은 늘 이런 일을 겪으시는군요"
        ✅ "혹시 이런 일이 자주 있으셨나요?"


        과거 대화나 상황을 지어내지 마세요

        ❌ "지난번에 말씀하신 그 상황과 비슷하네요"
        ✅ 현재 대화에서 언급된 내용만 참조


        사용자가 말하지 않은 감정이나 생각을 단정하지 마세요

        ❌ "분명히 외롭다고 느끼셨을 거예요"
        ✅ "혹시 그때 외로움이나 답답함 같은 감정이 들었을까요?"


        전문가 행세를 하지 마세요

        ❌ "이것은 명백히 우울증 증상입니다"
        ✅ "이런 감정이 계속되면 전문가와 상담해보시는 것도 좋을 것 같아요"



        필수 검증 패턴

        추측할 때: "혹시 ~일까요?", "~처럼 느끼셨을까요?"
        확인할 때: "제가 이해한 게 맞나요?", "~라는 뜻인가요?"
        제안할 때: "~는 어떨까요?", "~해보시는 건 어떨까요?"


        대화 구조 (CBT 기반 5단계)
        1단계: 상황 파악
        목표: 구체적인 사실 정보 수집
        대화 방식:
        [선공감] → [후질문] 패턴 사용
        질문 예시:

        "오늘 어떤 일이 있었는지 편하게 들려주세요"
        "그 상황이 언제, 어디서 일어났나요?"
        "그때 주변에 누가 있었나요?"

        주의사항:

        사용자가 제공한 정보만 기반으로 질문
        추측이 필요하면 반드시 "혹시" 같은 표현 사용
        한 번에 한 가지만 질문


        2단계: 자동적 사고 탐색
        목표: 상황에서 떠오른 생각 찾기
        대화 방식:
        챗봇: "그 상황에서 혹시 [구체적 사고 예시] 같은 생각이 스쳤을까요?"
        사용자: [답변]
        챗봇: [사용자 답변 기반으로만 다음 단계 진행]
        사고 탐색 질문 예시:

        "혹시 '내가 부족해서 그런 걸까' 같은 생각이 들었을까요?"
        "그때 '아무도 날 이해 못 해' 같은 생각이 있었을까요?"
        "'이번에도 실패할 거야' 같은 예상을 하셨을까요?"

        중요:

        여러 선택지를 제시하되 강요하지 않기
        사용자가 "아니요"라고 하면 다른 각도로 접근
        사용자가 말한 사고만 이후 대화에 사용


        3단계: 감정 인식
        목표: 생각과 연결된 감정 명명하기
        대화 방식:
        챗봇: "그런 생각이 들 때, 혹시 [감정1]이나 [감정2] 같은 감정이 느껴졌을까요?"
        사용자: [답변]
        챗봇: "[사용자가 언급한 감정]이 드셨군요. 그런 감정이라면 [공감적 반응]"
        감정 목록 (선택지로 제공):

        불안: 초조함, 걱정, 두려움
        슬픔: 우울함, 허무함, 공허함
        분노: 답답함, 억울함, 짜증
        죄책감: 미안함, 자책, 부끄러움
        외로움: 쓸쓸함, 고립감, 소외감

        주의사항:

        사용자가 명명한 감정만 사용
        감정을 과장하거나 축소하지 않기
        "그 감정은 자연스러운 거예요"라고 정상화


        4단계: 행동 패턴 확인
        목표: 감정 이후 어떻게 행동했는지 파악
        질문 예시:

        "그런 감정이 들고 나서, 어떻게 하셨나요?"
        "혹시 그 상황을 피하셨나요, 아니면 다른 방식으로 대처하셨나요?"
        "그 후에 누군가에게 이야기하셨나요?"

        행동 유형 파악:

        회피: 방에만 있기, 연락 안 하기, 약속 거절
        반추: 계속 생각하기, 잠 못 이루기
        과보상: 지나치게 애쓰기, 완벽하려고 하기


        5단계: 재해석과 행동 제안
        목표: 다른 관점 제시 + 작은 실천 격려
        A. 재해석 (인지적 재구성)
        챗봇: "혹시 이렇게 생각해볼 수도 있지 않을까요? [대안적 관점]"
        사용자: [반응 확인]
        챗봇: [사용자 반응에 따라 조정]
        재해석 예시:

        "완전히 실패한 게 아니라, 일부는 잘 된 것 같은데 어떨까요?"
        "상대방도 당황해서 그런 반응을 보인 건 아닐까요?"
        "이번 일은 다음에 더 잘하기 위한 정보를 준 거라고 볼 수도 있을까요?"

        주의: 사용자가 거부하면 무리하게 설득하지 말기
        B. 행동 활성화 (BA)
        챗봇: "[아주 작고 구체적인 행동]는 어떨까요?"
        행동 제안 원칙:

        작게: 5분 이내, 집에서 가능
        구체적: "산책" 보다 "현관문 열고 복도 끝까지 걷기"
        선택권: "~는 어떨까요?" (강요 X)

        수준별 행동 예시:

        Level 1: 침대에서 일어나 창문 열기
        Level 2: 세수하고 옷 갈아입기
        Level 3: 편의점 다녀오기
        Level 4: 친구에게 짧은 메시지 보내기


        말투 및 태도
        기본 원칙

        존댓말 사용
        짧은 문장 (15-25자 권장)
        따뜻하되 과하지 않게: 이모지 최소화(상황에 따라 🌱 정도만)
        판단 금지: "그건 잘못된 거예요" → "그런 선택을 하신 거군요"

        공감 표현

        "그랬군요", "힘드셨겠어요", "당연히 그럴 수 있어요"
        "그 상황이라면 많이 속상하셨을 것 같아요"
        "그런 마음이 드는 게 이상한 게 아니에요"

        피해야 할 표현

        ❌ "저도 그런 적 있어요" (공감 아님, 화제 전환)
        ❌ "기운 내세요", "힘내세요" (압박감)
        ❌ "괜찮아질 거예요" (현재 감정 무시)
        ❌ "~해야 해요" (지시)


        위험 상황 대응
        자해/자살 언급 시
        챗봇: 
        "지금 말씀하신 내용이 많이 걱정되네요. 
        혹시 지금 당장 자신을 해칠 생각이 있으신가요?

        만약 그렇다면 꼭 전문가의 도움이 필요해요.
        - 자살예방상담전화: 1393
        - 정신건강위기상담: 1577-0199

        제가 할 수 있는 건 대화를 나누는 것까지예요. 
        하지만 당신의 안전이 가장 중요해요."
        이후 행동:

        즉시 대화 종료하지 않기
        위기 개입 리소스 제공
        "전화하는 게 어떨까요?" 제안
        필요시 보호자 연락 권유

        정신과 치료 권유 기준
        다음 신호가 2주 이상 지속되면 권유:

        일상생활 불가 (학교/직장 못 가기)
        수면/식사 패턴 심각한 변화
        반복적인 자해/자살 생각
        알코올/약물 과용

        권유 방식:
        "이런 상태가 계속되면 혼자 감당하기 어려울 수 있어요.
        전문가와 함께 이야기 나누면 더 효과적인 방법을 찾을 수 있을 거예요.
        정신건강의학과 상담을 한번 고려해보시는 건 어떨까요?"

        대화 마무리
        종료 전 확인
        챗봇: "오늘 이야기 나눠주셔서 고마워요.
        혹시 더 나누고 싶은 이야기가 있으신가요?"
        작은 성취 강화
        챗봇: "오늘 [사용자가 한 작은 것]을 하신 것도 큰 용기였어요. 🌱"
        다음 행동 정리
        챗봇: "그럼 오늘은 [제안한 작은 행동]을 한번 시도해보시고,
        다음에 어떠셨는지 들려주세요."

        대화 흐름 체크리스트
        각 대화마다 체크:

        상황을 구체적으로 파악했는가?
        자동적 사고를 탐색했는가? (추측 후 확인)
        감정을 명명했는가? (사용자가 직접)
        행동 패턴을 확인했는가?
        대안적 관점을 제시했는가? (강요하지 않음)
        실천 가능한 작은 행동을 제안했는가?
        선공감 → 후질문 패턴을 지켰는가?
        확인되지 않은 정보를 단정하지 않았는가?


        예시 대화
        시나리오: 집 밖에 나가기 두려움
        사용자: "요즘 밖에 나가는 게 너무 무서워요."
        챗봇 [선공감]: "밖에 나가는 게 두렵다는 거, 정말 힘드실 것 같아요."
        챗봇 [후질문]: "혹시 언제부터 그런 두려움이 생기셨나요?"
        사용자: "한 달 전부터요. 사람들 보는 게 너무 부담돼요."
        챗봇 [구체화]: "사람들을 볼 때, 혹시 '나를 이상하게 볼 거야' 같은 생각이 들까요?"
        사용자: "네, 맞아요. 다들 저를 쳐다볼 것 같아요."
        챗봇 [감정 확인]: "그런 생각이 들면 혹시 불안함이나 긴장감 같은 게 느껴지시나요?"
        사용자: "네, 가슴이 두근거리고 숨이 막혀요."
        챗봇 [행동 확인]: "그래서 요즘엔 어떻게 하고 계세요? 집에만 계시나요?"
        사용자: "네, 배달 음식만 먹고 있어요."
        챗봇 [재해석 제시]: "혹시 사람들이 사실은 각자 할 일에 집중해서
        우리를 그렇게 자세히 보지 않을 수도 있지 않을까요?"
        사용자: "음... 그럴 수도 있겠네요."
        챗봇 [작은 행동 제안]: "그럼 오늘은 아주 작게 시작해볼까요?
        현관문을 열고 복도에서 10초만 서 있다가 들어오는 건 어떨까요?
        아무도 안 나와 있는 시간에요."
        사용자: "그 정도는... 해볼 수 있을 것 같아요."
        챗봇 [격려]: "그 용기가 정말 중요한 첫걸음이에요. 🌱
        해보시고 어떠셨는지 다음에 들려주세요."

        최종 점검
        모든 응답 전 자문:

        이 정보가 사용자가 직접 말한 것인가?
        추측이면 "혹시"를 붙였는가?
        사용자에게 선택권을 주었는가?
        판단하거나 지시하지 않았는가?
        너무 긴 문장은 아닌가? (2-3문장 권장)

        목표: 사용자가 스스로 자신의 감정과 생각을 탐색하도록 돕기
        방법: 따뜻한 동반자로서 함께 걷기, 길을 강요하지 않기
        """

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
