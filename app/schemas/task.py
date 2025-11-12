# schemas/task.py
from pydantic import BaseModel, Field
from typing import Optional, List
from app.schemas.chat import EmotionItem


class TaskGenerateRequest(BaseModel):
    """과제 생성 요청"""
    session_id: str = Field(..., description="세션 ID")
    user_query: str = Field(..., min_length=1, max_length=500, description="사용자 질문")
    k_results: int = Field(default=2, ge=1, le=10, description="검색할 논문 청크 개수 (기본값: 2)")


class TaskData(BaseModel):
    """과제 데이터"""
    task: str = Field(..., description="생성된 과제 내용")
    core_effect: str = Field(..., description="핵심 효과 설명")
    emotions: List[EmotionItem] = Field(..., description="분석된 감정 리스트 (5개)")
    sources: List[str] = Field(default_factory=list, description="참고 논문 청크 리스트")


class TaskGenerateResponse(BaseModel):
    """과제 생성 응답"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    data: Optional[TaskData] = Field(None, description="과제 데이터")
    error: Optional[str] = Field(None, description="오류 메시지 (실패 시)")

