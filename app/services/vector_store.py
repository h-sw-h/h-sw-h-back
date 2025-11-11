from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional
import os

class VectorStoreService:
    """
    RAG 기반 PDF 매뉴얼 서비스 (PostgreSQL + pgvector)
    - PDF 문서를 PostgreSQL 벡터 DB에 저장
    - 사용자 질문에 대해 문서 기반 답변 제공
    - 싱글톤 패턴으로 관리 (서버 전역 공유)
    """

    def __init__(self, openai_api_key: str, database_url: str, collection_name: str = "document_embeddings"):
        """
        초기화 함수
        - OpenAI API 키 설정
        - 임베딩 모델 설정 (text-embedding-3-small)
        - PostgreSQL 연결 설정
        """
        self.openai_api_key = openai_api_key
        self.database_url = database_url
        self.collection_name = collection_name

        # 직접 환경변수 등록
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # 임베딩 모델 설정 - 1536 차원
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

        self.vectorstore: Optional[PGVector] = None  # 벡터 DB
        self.qa_chain = None  # 질의응답 체인 (LCEL)
        self.retriever = None  # 문서 검색기

    def create_vectorstore(self, documents, batch_size: int = 100):
        """
        벡터 스토어 생성 (배치 처리)
        - 문서들을 임베딩으로 변환
        - PostgreSQL에 저장
        - 대용량 문서 처리를 위해 배치 단위로 처리
        """
        print(f"총 {len(documents)}개의 문서 청크를 임베딩합니다...")

        # 기존 컬렉션 삭제 후 새로 생성
        if self.vectorstore:
            # 기존 벡터스토어가 있으면 삭제
            try:
                PGVector.from_existing_index(
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    connection=self.database_url,
                ).delete_collection()
                print("기존 벡터 스토어 삭제 완료")
            except:
                pass

        # 배치 단위로 나누어 처리
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"배치 {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1} 처리 중... ({len(batch)}개)")

            if i == 0:
                # 첫 번째 배치: 새 벡터스토어 생성
                self.vectorstore = PGVector.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    connection=self.database_url,
                    pre_delete_collection=True  # 기존 컬렉션 삭제
                )
            else:
                # 이후 배치: 기존 벡터스토어에 추가
                self.vectorstore.add_documents(batch)

        print("임베딩 완료!")
        return self.vectorstore

    def load_vectorstore(self):
        """
        기존 벡터 스토어 불러오기
        - 이미 저장된 벡터 DB가 있으면 재사용
        """
        try:
            self.vectorstore = PGVector(
                collection_name=self.collection_name,
                connection=self.database_url,
                embeddings=self.embeddings,
            )
            # 테스트 쿼리로 데이터 존재 확인
            result = self.vectorstore.similarity_search("test", k=1)
            if result:
                print(f"벡터 스토어 로드 완료: {len(result)}개 문서 확인")
                return self.vectorstore
            else:
                print("벡터 스토어가 비어있습니다.")
                return None
        except Exception as e:
            print(f"벡터 스토어 로드 실패: {e}")
            return None

    def create_qa_chain(self, model_name: str = "gpt-4o-mini"):
        """
        QA 체인 생성 (LCEL - LangChain Expression Language)
        - LLM 모델 설정
        - 문서 검색 + 답변 생성 체인 구축
        """
        if not self.vectorstore:
            raise ValueError("벡터 스토어가 없습니다. create_vectorstore를 먼저 호출하세요.")

        # LLM 모델 설정
        llm = ChatOpenAI(
            model=model_name,
            temperature=0  # 일관된 답변을 위해 0으로 설정
        )

        # 친절한 한국어 프롬프트 (LCEL용)
        prompt_template = """너는 친절한 한국어 비서야.
아래 문서 내용을 참고해서 쉽고 자세하게 질문에 답해줘.
답변할 때는 존댓말을 사용하고, 핵심 내용을 먼저 말한 후 자세한 설명을 해줘.

문서 내용:
{context}

질문: {question}

답변:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Retriever 생성 (상위 3개 문서 검색)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # LCEL 체인 구성
        def format_docs(docs):
            """문서 리스트를 문자열로 변환"""
            return "\n\n".join(doc.page_content for doc in docs)

        # RAG 체인: 검색 → 포맷 → 프롬프트 → LLM → 파싱
        self.qa_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return self.qa_chain

    def query(self, question: str) -> dict:
        """
        질문에 답변하기 (LCEL)
        """
        if not self.qa_chain:
            raise ValueError("QA 체인이 없습니다. create_qa_chain을 먼저 호출하세요.")

        # LCEL 체인 실행 - 자동으로 문서 검색 + 답변 생성
        answer = self.qa_chain.invoke(question)

        # 소스 문서 별도 검색 (참조용)
        source_docs = self.retriever.invoke(question)

        return {
            "answer": answer,  # AI 답변
            "sources": [doc.page_content for doc in source_docs]  # 참조 문서
        }


# ============================================
# 싱글톤 패턴
# ============================================
_vector_store_instance: Optional[VectorStoreService] = None

def get_vector_store_service() -> VectorStoreService:
    """
    벡터 스토어 서비스 의존성 주입 (싱글톤)

    - 서버 시작 시 한 번 초기화
    - 전역 인스턴스 재사용
    """
    global _vector_store_instance

    if _vector_store_instance is None:
        from app.config import get_settings
        settings = get_settings()

        _vector_store_instance = VectorStoreService(
            openai_api_key=settings.openai_api_key,
            database_url=settings.database_url
        )

        # 기존 벡터 DB 자동 로드
        if _vector_store_instance.load_vectorstore():
            _vector_store_instance.create_qa_chain()
            print("✅ PDF 매뉴얼 벡터 스토어 로드 완료")
        else:
            print("⚠️  벡터 스토어가 비어있습니다. /api/chatbot/initialize를 호출하여 PDF를 로드하세요.")

    return _vector_store_instance

def reset_vector_store_service():
    """
    벡터 스토어 서비스 리셋 (재초기화용)

    - /api/chatbot/initialize에서 사용
    - 테스트에서 사용
    """
    global _vector_store_instance
    _vector_store_instance = None
