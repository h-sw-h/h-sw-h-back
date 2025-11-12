# services/diary_service.py
from typing import List, Dict, Optional
from datetime import datetime
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from sqlalchemy.orm import Session
from app.config import get_settings
import uuid

class DiaryService:
    """
    일기 관리 서비스 (PostgreSQL + pgvector)
    - 일기 저장 (벡터 임베딩 포함)
    - 유사 일기 검색 (RAG)
    - user_id별 네임스페이스 분리
    """

    def __init__(self):
        settings = get_settings()
        self.openai_api_key = settings.openai_api_key
        self.database_url = settings.database_url

        # 임베딩 모델 (text-embedding-3-small, 1536 차원)
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

        # 일기 전용 컬렉션
        self.collection_name = "user_diaries"

    def save_diary(
        self,
        user_id: str,
        diary_content: str,
        alternative_perspective: str,
        message_count: int
    ) -> str:
        """
        일기 저장

        Args:
            user_id: 사용자 ID
            diary_content: 일기 본문

        Returns:
            diary_id: 저장된 일기 ID
        """
        diary_id = str(uuid.uuid4())
        now = datetime.now()

        # 날짜 계산: 06:00 이전이면 전날로 기록
        if now.hour < 6:
            diary_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            diary_date = now.strftime("%Y-%m-%d")

        # 메타데이터 구성 (필수 필드만)
        doc_metadata = {
            "user_id": user_id,
            "diary_id": diary_id,
            "diary_date": diary_date,  # 일기 날짜 (06:00 기준)
            "created_at": now.isoformat(),  # 실제 생성 시각
            "alternative_perspective": alternative_perspective,
            "message_count": message_count
        }

        # Document 객체 생성
        document = Document(
            page_content=diary_content,
            metadata=doc_metadata
        )

        # PGVector에 저장
        vectorstore = PGVector.from_documents(
            documents=[document],
            embedding=self.embeddings,
            collection_name=self.collection_name,
            connection=self.database_url,
            pre_delete_collection=False  # 기존 일기 유지
        )

        return diary_id

    def search_similar_diaries(
        self,
        user_id: str,
        query: str,
        k: int = 3
    ) -> List[Dict]:
        """
        유사 일기 검색 (RAG 리트리벌)

        Args:
            user_id: 사용자 ID (네임스페이스 필터)
            query: 검색 쿼리 (사용자 메시지)
            k: 반환할 일기 개수 (기본 3개)

        Returns:
            유사 일기 리스트 (content + metadata)
        """
        try:
            # PGVector 인스턴스 생성
            vectorstore = PGVector(
                collection_name=self.collection_name,
                connection=self.database_url,
                embeddings=self.embeddings
            )

            # 유사도 검색 (user_id 필터링)
            # PGVector는 metadata 필터를 지원합니다
            results = vectorstore.similarity_search(
                query,
                k=k,
                filter={"user_id": user_id}  # user_id로 필터링
            )

            # 결과 포맷팅
            diaries = []
            for doc in results:
                diaries.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

            return diaries

        except Exception as e:
            print(f"일기 검색 실패: {e}")
            return []

    def get_user_diary_count(self, user_id: str) -> int:
        """
        사용자의 총 일기 개수 조회

        Args:
            user_id: 사용자 ID

        Returns:
            일기 개수
        """
        try:
            vectorstore = PGVector(
                collection_name=self.collection_name,
                connection=self.database_url,
                embeddings=self.embeddings
            )

            # user_id로 필터링하여 모든 일기 검색
            results = vectorstore.similarity_search(
                "",  # 빈 쿼리
                k=1000,  # 충분히 큰 수
                filter={"user_id": user_id}
            )

            return len(results)

        except:
            return 0


# 전역 일기 서비스 인스턴스
_diary_service: Optional[DiaryService] = None

def get_diary_service() -> DiaryService:
    """
    일기 서비스 의존성 주입
    """
    global _diary_service
    if _diary_service is None:
        _diary_service = DiaryService()
    return _diary_service
