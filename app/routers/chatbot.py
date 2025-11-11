# 챗봇 API 라우터
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.chatbot import QuestionRequest, ChatResponse, StatusResponse
from app.services.vector_store import (
    VectorStoreService,
    get_vector_store_service,
    reset_vector_store_service
)
from app.utils.pdf_loader import PDFProcessor
from app.config import get_settings
import os
import traceback
from app.routers.auth import get_current_user_id

router = APIRouter()


# ============================================
# API 엔드포인트
# ============================================

@router.post("/initialize", response_model=StatusResponse, summary="PDF 매뉴얼 재초기화")
async def initialize_chatbot():
    """
    PDF 매뉴얼 재초기화 API (관리자용)

    - data 폴더의 모든 PDF 문서 로드
    - 기존 벡터 DB 삭제 후 재생성
    - QA 체인 구축

    **주의:** 일반적으로는 서버 시작 시 자동 로드되므로 이 API는 불필요합니다.
    PDF 파일이 추가/변경되었을 때만 호출하세요.
    """
    try:
        # 1. 설정 확인
        settings = get_settings()

        if not settings.openai_api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY가 설정되지 않았습니다."
            )

        if not settings.database_url:
            raise HTTPException(
                status_code=500,
                detail="DATABASE_URL이 설정되지 않았습니다."
            )

        # 2. data 디렉토리 확인
        data_dir = os.path.join(os.getcwd(), "data")

        if not os.path.exists(data_dir):
            raise HTTPException(
                status_code=404,
                detail=f"data 디렉토리를 찾을 수 없습니다: {data_dir}"
            )

        # 3. 기존 싱글톤 인스턴스 리셋
        reset_vector_store_service()

        # 4. 새 벡터 스토어 서비스 생성
        vector_store = VectorStoreService(
            openai_api_key=settings.openai_api_key,
            database_url=settings.database_url
        )

        # 5. PDF 로드 및 벡터 DB 재생성
        pdf_processor = PDFProcessor(data_dir=data_dir)
        documents, loaded_files = pdf_processor.load_and_split()

        vector_store.create_vectorstore(documents)
        vector_store.create_qa_chain()

        # 6. 싱글톤 인스턴스에 등록
        from app.services import vector_store as vs_module
        vs_module._vector_store_instance = vector_store

        return StatusResponse(
            success=True,
            message=f"PDF 매뉴얼 재초기화 완료! {len(loaded_files)}개 파일, {len(documents)}개 문서 청크",
            data={
                "status": "reinitialized",
                "initialized": True,
                "document_count": len(documents),
                "loaded_files": loaded_files
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"초기화 실패: {str(e)}\n\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"초기화 실패: {str(e)}")


@router.post("/chat", response_model=ChatResponse, summary="PDF 매뉴얼 질의응답")
async def chat(
    request: QuestionRequest,
    user_id: str = Depends(get_current_user_id),
    vector_store: VectorStoreService = Depends(get_vector_store_service)
):
    """
    PDF 매뉴얼 기반 질의응답 API

    - 사용자 질문에 대해 PDF 문서 기반 답변 제공
    - 상위 3개 관련 문서 검색
    - GPT-4o-mini로 답변 생성
    """
    try:
        result = vector_store.query(request.question)

        return ChatResponse(
            success=True,
            message="답변이 생성되었습니다.",
            data={
                "question": request.question,
                "answer": result["answer"],
                "sources": result.get("sources"),
                "user_id": user_id
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"답변 생성 실패: {str(e)}"
        )
