from pydantic import BaseModel, Field
from typing import List, Optional, Any, Generic, TypeVar

# ============================================
# 공통 응답 포맷
# ============================================
T = TypeVar('T')

class ApiResponse(BaseModel, Generic[T]):
    """
    통일된 API 응답 형식
    - success: 성공 여부
    - message: 응답 메시지
    - data: 실제 데이터 (제네릭)
    """
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    data: Optional[T] = Field(None, description="응답 데이터")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "요청이 성공했습니다",
                "data": {}
            }
        }


# ============================================
# Request 스키마
# ============================================
class QuestionRequest(BaseModel):
    question: str = Field(..., description="사용자의 질문")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "아무것도 하기 싫을 때 어떻게 시작하면 좋을까?"
            }
        }


class InitializeRequest(BaseModel):
    """초기화 요청 - 현재는 본문 없이 data 폴더의 모든 PDF를 자동 로드"""
    pass


# ============================================
# Response Data 스키마
# ============================================
class ChatData(BaseModel):
    """챗봇 대화 응답 데이터"""
    question: str = Field(..., description="사용자의 질문")
    answer: str = Field(..., description="AI의 답변")
    sources: Optional[List[str]] = Field(None, description="참조한 문서 출처")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "우울감을 회복하기위한 지원 프로그램에는 어떤 것들이 있나요?",
                "answer": "우울감을 갖는 분들을 위한 지원 프로그램은...",
                "sources": ["지원 프로그램 안내...", "상담 서비스 정보..."]
            }
        }


class StatusData(BaseModel):
    """상태 응답 데이터"""
    status: str = Field(..., description="상태")
    initialized: bool = Field(..., description="초기화 여부")
    document_count: Optional[int] = Field(None, description="문서 청크 수")
    loaded_files: Optional[List[str]] = Field(None, description="로드된 PDF 파일 목록")


class InitializeData(BaseModel):
    """초기화 응답 데이터"""
    document_count: int = Field(..., description="로드된 문서 청크 수")
    vector_db_path: str = Field(..., description="벡터 DB 경로")
    loaded_files: List[str] = Field(..., description="로드된 PDF 파일 목록")


# ============================================
# 구체적인 Response 타입들
# ============================================
class ChatResponse(ApiResponse[ChatData]):
    """챗봇 대화 응답"""
    pass


class StatusResponse(ApiResponse[StatusData]):
    """상태 확인 응답"""
    pass


class InitializeResponse(ApiResponse[InitializeData]):
    """초기화 응답"""
    pass
