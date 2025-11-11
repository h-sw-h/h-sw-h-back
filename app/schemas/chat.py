# schemas/chat.py
from pydantic import BaseModel, Field
from typing import Optional, List

class SessionCreateRequest(BaseModel):
    """세션 생성 요청"""
    pass  # user_id는 JWT에서 추출

class SessionCreateResponse(BaseModel):
    """세션 생성 응답"""
    success: bool
    message: str
    data: dict

class ChatMessageRequest(BaseModel):
    """채팅 메시지 요청"""
    session_id: str = Field(..., description="세션 ID")
    message: str = Field(..., min_length=1, max_length=2000, description="사용자 메시지")

class ChatMessageResponse(BaseModel):
    """채팅 메시지 응답"""
    success: bool
    message: str
    data: dict  # answer, similar_diaries 등

class SessionEndRequest(BaseModel):
    """세션 종료 요청"""
    session_id: str = Field(..., description="종료할 세션 ID")

class SessionEndResponse(BaseModel):
    """세션 종료 응답"""
    success: bool
    message: str
    data: dict  # diary_id, diary_content 등
