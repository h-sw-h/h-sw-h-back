# 챗봇 API 라우터
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.chatbot import QuestionRequest, ChatResponse, StatusResponse
from app.services.vector_store import VectorStoreService
from app.utils.pdf_loader import PDFProcessor
from app.config import get_settings
import os
import traceback
from app.routers.auth import get_current_user_id

router = APIRouter()

# 전역 벡터 스토어 서비스 (싱글톤 패턴)
vector_store_service: VectorStoreService = None


# ============================================
# Dependency (의존성 주입)
# ============================================
def get_vector_store():
    """벡터 스토어 의존성 - 초기화 여부 확인"""
    global vector_store_service
    if vector_store_service is None:
        raise HTTPException(
            status_code=503,
            detail="챗봇이 초기화되지 않았습니다. /initialize를 먼저 호출해주세요."
        )
    return vector_store_service


# ============================================
# API 엔드포인트
# ============================================

@router.post("/initialize", response_model=StatusResponse, summary="챗봇 초기화")
async def initialize_chatbot():
    """
    챗봇 초기화 API
    - data 폴더의 모든 PDF 문서 로드
    - 벡터 DB 생성
    - QA 체인 구축
    """
    global vector_store_service

    try:
        # 1. OpenAI API 키 확인
        settings = get_settings()
        openai_api_key = settings.openai_api_key
        if not openai_api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY가 설정되지 않았습니다."
            )

        # 2. data 디렉토리 설정
        data_dir = os.path.join(os.getcwd(), "data")

        if not os.path.exists(data_dir):
            raise HTTPException(
                status_code=404,
                detail=f"data 디렉토리를 찾을 수 없습니다: {data_dir}"
            )

        # 3. 벡터 스토어 서비스 생성
        vector_store_service = VectorStoreService(openai_api_key=openai_api_key)

        # 4. 기존 벡터 DB가 있으면 로드
        if vector_store_service.load_vectorstore():
            vector_store_service.create_qa_chain()
            return StatusResponse(
                success=True,
                message="기존 벡터 DB에서 챗봇을 초기화했습니다.",
                data={
                    "status": "loaded_existing",
                    "initialized": True,
                    "document_count": None
                }
            )

        # 5. 새로 PDF 로드 및 벡터 DB 생성
        pdf_processor = PDFProcessor(data_dir=data_dir)
        documents, loaded_files = pdf_processor.load_and_split()

        vector_store_service.create_vectorstore(documents)
        vector_store_service.create_qa_chain()

        return StatusResponse(
            success=True,
            message=f"챗봇 초기화 완료! {len(loaded_files)}개 파일, {len(documents)}개 문서 청크",
            data={
                "status": "initialized",
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


@router.post("/chat", response_model=ChatResponse, summary="챗봇 대화")
async def chat(
    request: QuestionRequest,
    user_id: str = Depends(get_current_user_id) ,
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """
    챗봇 질의응답 API
    - 사용자 질문에 대해 문서 기반 답변 제공
    """
    try:
        # todo: user_id 활용한 맞춤형 응답 기능 추가 예정
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
