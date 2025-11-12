# services/task_generator_service.py
"""
Emotion 기반 과제 생성 서비스
- 세션 기반 감정 분석을 통해 5개 emotion과 score 추출
- 논문 청크 검색 및 LangChain을 사용하여 과제(task)와 핵심 효과(core_effect) 생성
"""

from typing import List, Dict, Any, Optional
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.config import get_settings
from app.utils.pdf_to_vector import get_pg_conn, load_embedder
from app.services.chat_session import ChatSessionManager, get_session_manager
from app.services.emotion_service import EmotionService, get_emotion_service


class TaskGeneratorService:
    """
    Emotion 기반 과제 생성 서비스
    - 세션 기반 감정 분석을 통해 5개 emotion과 score 추출
    - PostgreSQL + pgvector에서 논문 청크 검색
    - LangChain으로 과제 생성
    """
    
    def __init__(
        self,
        table_name: str = "thesis_chunks",
        session_manager: Optional[ChatSessionManager] = None,
        emotion_service: Optional[EmotionService] = None
    ):
        """
        초기화
        
        Args:
            table_name: 검색할 테이블 이름 (기본값: thesis_chunks)
            session_manager: 세션 관리자 (선택, 없으면 싱글톤 사용)
            emotion_service: 감정 분석 서비스 (선택, 없으면 싱글톤 사용)
        """
        settings = get_settings()
        self.openai_api_key = settings.openai_api_key
        self.database_url = settings.database_url
        self.table_name = table_name
        
        # 세션 관리자 및 감정 분석 서비스
        self.session_manager = session_manager or get_session_manager()
        self.emotion_service = emotion_service or get_emotion_service()
        
        # 임베딩 모델 로드
        self.embedder, self.dim = load_embedder()
        
        # LLM 모델 설정
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=self.openai_api_key,
            temperature=0.7
        )
        
        # System Prompt 설정
        self.system_prompt = (
            "당신은 우울증을 겪는 사용자에게 따뜻한 공감과 함께 실질적인 행동 과제를 제시하는 심리학 전문가입니다. "
            "답변은 친절하고 격려하는 구어체로 작성되어야 합니다. "
            "사용자가 제공한 논문(청크) 내용을 바탕으로, "
            "'8개의 핵심 치료 기법' 항목을 기반으로 사용자에게 구체적이고 실행 가능한 과제(Task)를 제시해야 합니다. "
            "또한, 이 과제 수행으로 얻을 수 있는 '핵심 효과(Core Effect)'를 논문 내용과 연결하여 설명하십시오.\n\n"
            "출력은 반드시 다음 JSON 형식만을 따르십시오.\n\n"
            "{\n"
            '  "task": "사용자가 실천할 수 있는 구체적인 행동 과제",\n'
            '  "core_effect": "해당 과제를 통해 기대되는 핵심 심리적 효과"\n'
            "}"
        )
        
        # 프롬프트 템플릿 생성 (LangChain)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """사용자 질문: {user_query}

사용자의 감정 상태:
{emotions_info}

참고 논문 내용:
{context_text}

위 정보를 바탕으로 JSON 형식으로 과제와 핵심 효과를 생성해주세요.""")
        ])
        
        # LangChain 체인 구성 (LCEL)
        self.chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )
    
    def search_thesis_chunks(
        self,
        query: str,
        k: int = 2
    ) -> str:
        """
        PostgreSQL + pgvector에서 논문 청크 검색
        
        Args:
            query: 검색 쿼리 텍스트
            k: 반환할 상위 K개 결과
        
        Returns:
            검색된 청크들을 결합한 텍스트
        """
        try:
            conn = get_pg_conn(self.database_url)
        except Exception as e:
            print(f"❌ PostgreSQL 연결 오류: {e}")
            return ""
        
        try:
            # 테이블 존재 확인
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                """, (self.table_name,))
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    print(f"❌ 오류: 테이블 '{self.table_name}'을(를) 찾을 수 없습니다.")
                    conn.close()
                    return ""
            
            # 쿼리 텍스트를 임베딩으로 변환
            query_embedding = self.embedder.embed_query(query)
            vec_literal = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"
            
            # 유사도 검색 실행
            sql = f"""
            SELECT 
                content,
                source,
                chunk_index
            FROM {self.table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """
            
            with conn.cursor() as cur:
                cur.execute(sql, (vec_literal, k))
                rows = cur.fetchall()
            
            # 결과 포맷팅
            if not rows:
                conn.close()
                return ""
            
            combined_text = ""
            for i, (content, source, chunk_index) in enumerate(rows):
                combined_text += f"[청크 {i+1}]\n{content}\n\n"
            
            conn.close()
            return combined_text
            
        except Exception as e:
            print(f"❌ 검색 실행 오류: {e}")
            import traceback
            traceback.print_exc()
            conn.close()
            return ""
    
    def format_emotions(
        self,
        emotions: List[Dict[str, Any]]
    ) -> str:
        """
        5개 emotion과 score를 포맷팅
        
        Args:
            emotions: [{"emotion": "슬픔", "score": 0.85}, ...] 형식의 리스트
        
        Returns:
            포맷팅된 감정 정보 문자열
        """
        if not emotions:
            return "감정 정보가 없습니다."
        
        emotion_lines = []
        for i, emo in enumerate(emotions[:5], 1):  # 최대 5개만 사용
            emotion = emo.get("emotion", "알 수 없음")
            score = emo.get("score", 0.0)
            emotion_lines.append(f"{i}. {emotion}: {score:.2f}")
        
        return "\n".join(emotion_lines)
    
    def analyze_session_emotions(self, session_id: str) -> List[Dict[str, Any]]:
        """
        세션 기반 감정 분석 수행
        
        Args:
            session_id: 세션 ID
        
        Returns:
            5개 emotion과 score 리스트
        """
        # 1. 세션 존재 확인
        session_info = self.session_manager.get_session_info(session_id)
        if not session_info:
            print(f"❌ 세션을 찾을 수 없습니다: {session_id}")
            return []
        
        # 2. 전체 대화 내역 가져오기
        full_conversation = self.session_manager.get_full_conversation(session_id)
        if not full_conversation:
            print(f"❌ 세션에 대화 내용이 없습니다: {session_id}")
            return []
        
        # 3. 사용자 메시지만 필터링하여 한 줄 텍스트로 합치기
        combined_text = self.emotion_service.combine_conversation_text(full_conversation)
        if not combined_text or not combined_text.strip():
            print(f"❌ 사용자 메시지가 없습니다: {session_id}")
            return []
        
        # 4. 감정 분석 (상위 5개)
        print(f"😊 세션 '{session_id}'의 감정 분석 중...")
        emotion_results = self.emotion_service.analyze_emotions(combined_text, top_k=5)
        
        if not emotion_results:
            print(f"⚠️  감정 분석 결과가 없습니다: {session_id}")
            return []
        
        print(f"✅ 감정 분석 완료: {len(emotion_results)}개 감정 감지")
        return emotion_results
    
    def generate_task_from_session(
        self,
        session_id: str,
        user_query: str,
        k_results: int = 2
    ) -> Dict[str, Any]:
        """
        세션 기반으로 과제 생성 (전체 체인 - LangChain)
        
        Args:
            session_id: 세션 ID
            user_query: 사용자 질문
            k_results: 검색할 논문 청크 개수
        
        Returns:
            {
                "task": "과제 내용",
                "core_effect": "핵심 효과",
                "sources": ["청크1", "청크2", ...],
                "emotions": [...]
            }
        """
        # 1. 세션 기반 감정 분석
        emotions = self.analyze_session_emotions(session_id)
        
        if not emotions:
            return {
                "task": "",
                "core_effect": "",
                "sources": [],
                "emotions": [],
                "error": "세션에서 감정을 분석할 수 없습니다."
            }
        
        # 2. 논문 청크 검색
        print(f"🔎 쿼리: '{user_query}'에 대해 논문 청크 {k_results}개를 검색합니다...")
        context_text = self.search_thesis_chunks(user_query, k=k_results)
        
        if not context_text:
            return {
                "task": "",
                "core_effect": "",
                "sources": [],
                "emotions": emotions,
                "error": "논문 청크를 찾을 수 없습니다."
            }
        
        # 3. 감정 정보 포맷팅
        emotions_info = self.format_emotions(emotions)
        
        # 4. LangChain 체인 실행
        print("🤖 LangChain으로 과제 생성 중...")
        try:
            response = self.chain.invoke({
                "user_query": user_query,
                "emotions_info": emotions_info,
                "context_text": context_text
            })
            
            # JSON 파싱 시도
            try:
                # JSON 부분만 추출 (마크다운 코드 블록 제거)
                json_text = response.strip()
                if json_text.startswith("```json"):
                    json_text = json_text[7:]
                if json_text.startswith("```"):
                    json_text = json_text[3:]
                if json_text.endswith("```"):
                    json_text = json_text[:-3]
                json_text = json_text.strip()
                
                result = json.loads(json_text)
                
                return {
                    "task": result.get("task", ""),
                    "core_effect": result.get("core_effect", ""),
                    "sources": context_text.split("\n\n")[:k_results],
                    "emotions": emotions
                }
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 원본 텍스트 반환
                print("⚠️  JSON 파싱 실패, 원본 텍스트 반환")
                return {
                    "task": response,
                    "core_effect": "",
                    "sources": context_text.split("\n\n")[:k_results],
                    "emotions": emotions,
                    "raw_response": response
                }
                
        except Exception as e:
            print(f"❌ 과제 생성 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                "task": "",
                "core_effect": "",
                "sources": [],
                "emotions": emotions,
                "error": str(e)
            }


# ============================================
# 싱글톤 패턴
# ============================================
_task_generator_instance: Optional[TaskGeneratorService] = None

def get_task_generator_service() -> TaskGeneratorService:
    """
    Task Generator 서비스 의존성 주입 (싱글톤)
    """
    global _task_generator_instance
    
    if _task_generator_instance is None:
        _task_generator_instance = TaskGeneratorService()
    
    return _task_generator_instance

def reset_task_generator_service():
    """
    Task Generator 서비스 리셋 (재초기화용)
    """
    global _task_generator_instance
    _task_generator_instance = None

