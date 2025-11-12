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

class DiaryEntry(BaseModel):
    """일기 항목"""
    content: str = Field(..., description="일기 내용")
    metadata: dict = Field(..., description="일기 메타데이터")
    diary_date: str = Field(..., description="일기 날짜 (YYYY-MM-DD, 06:00 기준)")

class WeeklyDiariesData(BaseModel):
    """일주일치 일기 조회 데이터"""
    diaries: List[DiaryEntry] = Field(..., description="일기 리스트 (날짜순 정렬, 최신순)")
    count: int = Field(..., description="일기 개수")
    days: int = Field(..., description="조회한 일수")

class WeeklyDiariesResponse(BaseModel):
    """일주일치 일기 조회 응답"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    data: WeeklyDiariesData = Field(..., description="일기 조회 데이터")

class DiaryByDateResponse(BaseModel):
    """특정 날짜 일기 조회 응답"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    data: Optional[DiaryEntry] = Field(None, description="일기 데이터 (없으면 None)")

class ChatHistoryMessage(BaseModel):
    """채팅 메시지"""
    role: str = Field(..., description="메시지 역할 (user 또는 assistant)")
    content: str = Field(..., description="메시지 내용")
    timestamp: Optional[str] = Field(None, description="메시지 타임스탬프")

class ChatHistoryData(BaseModel):
    """채팅 내역 데이터"""
    session_id: str = Field(..., description="세션 ID")
    messages: List[ChatHistoryMessage] = Field(..., description="메시지 리스트")
    message_count: int = Field(..., description="총 메시지 개수")

class ChatHistoryResponse(BaseModel):
    """채팅 내역 조회 응답"""
    success: bool = Field(..., description="성공 여부")
    data: ChatHistoryData = Field(..., description="채팅 내역 데이터")

class EmotionItem(BaseModel):
    """감정 항목"""
    emotion: str = Field(..., description="감정 이름 (KOTE 43 레이블)")
    score: float = Field(..., ge=0.0, le=1.0, description="감정 점수 (0.0 ~ 1.0)")
    threshold: float = Field(..., ge=0.0, le=1.0, description="감정 활성화 임계값")
    is_active: bool = Field(..., description="임계값을 넘었는지 여부")

class EmotionAnalysisData(BaseModel):
    """감정 분석 데이터"""
    session_id: str = Field(..., description="세션 ID")
    combined_text: str = Field(..., description="합쳐진 대화 텍스트")
    emotions: List[EmotionItem] = Field(..., description="상위 5개 감정 리스트 (점수 내림차순)")
    message_count: int = Field(..., description="분석에 사용된 사용자 메시지 개수")

class EmotionAnalysisResponse(BaseModel):
    """감정 분석 응답"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    data: EmotionAnalysisData = Field(..., description="감정 분석 데이터")
