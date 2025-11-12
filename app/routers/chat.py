# routers/chat.py
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.chat import (
    SessionCreateRequest,
    SessionCreateResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    SessionEndRequest,
    SessionEndResponse,
    ChatHistoryResponse,
    ChatHistoryData,
    ChatHistoryMessage,
    EmotionAnalysisResponse,
    EmotionAnalysisData,
    EmotionItem
)
from app.services.chat_session import get_session_manager, ChatSessionManager
from app.services.diary_service import get_diary_service, DiaryService
from app.services.chat_orchestrator import get_chat_orchestrator, ChatOrchestrator
from app.services.vector_store import VectorStoreService, get_vector_store_service
from app.services.emotion_service import get_emotion_service, EmotionService
from app.routers.auth import get_current_user_id
import traceback

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post("/session/create", response_model=SessionCreateResponse, summary="채팅 세션 시작")
async def create_session(
    user_id: str = Depends(get_current_user_id),
    session_manager: ChatSessionManager = Depends(get_session_manager)
):
    """
    채팅 세션 생성 (멱등성 보장)

    - **멱등성**: session_id = user_id (항상 동일한 세션)
    - **TTL**: 30분 자동 만료 (활동 시마다 연장)

    **동작:**
    1. JWT 토큰에서 user_id 추출
    2. user_id를 session_id로 사용
    3. Redis에 저장 및 TTL 설정
    """
    try:
        # 세션 생성 (user_id = session_id)
        session_id = session_manager.create_session(user_id)
        session_info = session_manager.get_session_info(session_id)
        is_new = session_info.get("message_count", "0") == "0"

        return SessionCreateResponse(
            success=True,
            message="세션이 생성되었습니다." if is_new else "기존 세션이 반환되었습니다.",
            data={
                "session_id": session_id,
                "user_id": user_id,
                "is_new": is_new,
                "message_count": int(session_info.get("message_count", 0)),
                "ttl_minutes": 30
            }
        )

    except Exception as e:
        print(f"세션 생성 실패: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"세션 생성 실패: {str(e)}")


@router.post("/message", response_model=ChatMessageResponse, summary="메시지 전송")
async def send_message(
    request: ChatMessageRequest,
    user_id: str = Depends(get_current_user_id),
    vector_store: VectorStoreService = Depends(get_vector_store_service)
):
    """
    채팅 메시지 전송

    **플로우:**
    1. 세션 검증
    2. 사용자 메시지 저장
    3. 과거 일기 검색 (RAG)
    4. 컨텍스트 구성 (시스템 프롬프트 + 과거 일기 + 최근 대화)
    5. OpenAI 모델 호출
    6. 응답 저장
    7. 응답 반환
    """
    try:
        # Orchestrator 생성 (vector_store 포함)
        orchestrator = get_chat_orchestrator(vector_store=vector_store)

        # 세션 검증 (user_id 확인)
        session_info = orchestrator.session_manager.get_session_info(request.session_id)

        if not session_info:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        if session_info.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="세션 접근 권한이 없습니다.")

        # 메시지 처리 (전체 플로우)
        result = orchestrator.process_message(
            session_id=request.session_id,
            user_message=request.message
        )

        return ChatMessageResponse(
            success=True,
            message="응답이 생성되었습니다.",
            data={
                "session_id": request.session_id,
                "user_message": request.message,
                "assistant_response": result["answer"],
                "referenced_diaries": result.get("similar_diaries")
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"메시지 처리 실패: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"메시지 처리 실패: {str(e)}")


@router.post("/session/end", response_model=SessionEndResponse, summary="세션 종료 및 일기 생성")
async def end_session(
    request: SessionEndRequest,
    user_id: str = Depends(get_current_user_id),
    vector_store: VectorStoreService = Depends(get_vector_store_service),
    diary_service: DiaryService = Depends(get_diary_service)
):
    """
    세션 종료 및 일기 생성

    **플로우:**
    1. 세션 검증
    2. 전체 대화 요약 (CBT 구조)
    3. 일기 저장 (Vector DB 영속화)
    4. 세션 삭제 (Redis 정리)
    5. 일기 ID 및 내용 반환
    """
    try:
        # Orchestrator 생성 (vector_store 포함)
        orchestrator = get_chat_orchestrator(vector_store=vector_store)

        # 세션 검증
        session_info = orchestrator.session_manager.get_session_info(request.session_id)

        if not session_info:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        if session_info.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="세션 접근 권한이 없습니다.")

        # 대화 요약 → 일기 생성
        diary_result = orchestrator.summarize_conversation_to_diary(request.session_id)

        if not diary_result or not diary_result.get("diary_text"):
            raise HTTPException(status_code=400, detail="대화 내용이 없어 일기를 생성할 수 없습니다.")

        # 일기 저장 (Vector DB)
        diary_id = diary_service.save_diary(
            user_id=user_id,
            diary_content=diary_result["diary_text"],
            alternative_perspective=diary_result.get("alternative_perspective", ""),
            message_count=session_info.get("message_count", 0)
        )

        # 세션 삭제
        # todo: 세션을 나중에 참조할 수도 있어서 일단 삭제하지 않음 (6AM 에 삭제 예정)
        # orchestrator.session_manager.delete_session(request.session_id)

        return SessionEndResponse(
            success=True,
            message="세션이 종료되고 일기가 생성되었습니다.",
            data={
                "diary_id": diary_id,
                "diary_content": diary_result["diary_text"],
                "alternative_perspective": diary_result.get("alternative_perspective", ""),
                "message_count": session_info.get("message_count", 0)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"세션 종료 실패: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"세션 종료 실패: {str(e)}")


@router.get("/session/{session_id}/history", response_model=ChatHistoryResponse, summary="세션 채팅 내역 조회")
async def get_chat_history(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    session_manager: ChatSessionManager = Depends(get_session_manager)
):
    """
    특정 세션의 모든 채팅 내역 조회

    **응답:**
    - session_id: 세션 ID
    - user_id: 사용자 ID
    - messages: 메시지 리스트 (role, content, timestamp)
    - message_count: 총 메시지 개수
    """
    try:
        # 세션 검증
        session_info = session_manager.get_session_info(session_id)

        if not session_info:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        if session_info.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="세션 접근 권한이 없습니다.")

        # 전체 대화 내역 가져오기
        full_conversation = session_manager.get_full_conversation(session_id)

        # ChatHistoryMessage 객체로 변환
        messages = [
            ChatHistoryMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg.get("timestamp")
            )
            for msg in full_conversation
        ]

        return ChatHistoryResponse(
            success=True,
            data=ChatHistoryData(
                session_id=session_id,
                messages=messages,
                message_count=len(messages)
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"채팅 내역 조회 실패: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"채팅 내역 조회 실패: {str(e)}")


@router.get("/session/{session_id}/emotions", response_model=EmotionAnalysisResponse, summary="감정 분석")
async def analyze_emotions(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    session_manager: ChatSessionManager = Depends(get_session_manager),
    emotion_service: EmotionService = Depends(get_emotion_service)
):
    """
    세션의 사용자 대화 텍스트를 분석하여 상위 5개 감정 반환

    **플로우:**
    1. 세션 검증
    2. 전체 대화 내역 가져오기
    3. 사용자 메시지만 필터링하여 한 줄 텍스트로 합치기
    4. kote-bert-ml 모델에 입력하여 감정 분석
    5. 상위 5개 감정 반환 (모든 감정 포함)

    **응답:**
    - session_id: 세션 ID
    - combined_text: 합쳐진 대화 텍스트
    - emotions: 상위 5개 감정 리스트 (점수 내림차순)
      - emotion: 감정 이름 (KOTE 43 레이블)
      - score: 감정 점수 (0.0 ~ 1.0)
      - threshold: 감정 활성화 임계값
      - is_active: 임계값을 넘었는지 여부
    - message_count: 분석에 사용된 사용자 메시지 개수
    """
    try:
        # 세션 검증
        session_info = session_manager.get_session_info(session_id)

        if not session_info:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

        if session_info.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="세션 접근 권한이 없습니다.")

        # 전체 대화 내역 가져오기
        full_conversation = session_manager.get_full_conversation(session_id)

        if not full_conversation:
            raise HTTPException(status_code=400, detail="분석할 대화 내용이 없습니다.")

        # 사용자 메시지만 필터링하여 한 줄 텍스트로 합치기
        combined_text = emotion_service.combine_conversation_text(full_conversation)

        if not combined_text or not combined_text.strip():
            raise HTTPException(status_code=400, detail="사용자 메시지가 없습니다.")

        # 감정 분석 (상위 5개)
        emotion_results = emotion_service.analyze_emotions(combined_text, top_k=5)

        if not emotion_results:
            raise HTTPException(status_code=500, detail="감정 분석 모델이 로드되지 않았습니다.")

        # 사용자 메시지 개수 계산
        user_message_count = sum(1 for msg in full_conversation if msg.get("role") == "user")

        # EmotionItem 객체로 변환
        emotions = [
            EmotionItem(
                emotion=item["emotion"],
                score=item["score"],
                threshold=item["threshold"],
                is_active=item["is_active"]
            )
            for item in emotion_results
        ]

        return EmotionAnalysisResponse(
            success=True,
            message="감정 분석이 완료되었습니다.",
            data=EmotionAnalysisData(
                session_id=session_id,
                combined_text=combined_text,
                emotions=emotions,
                message_count=user_message_count
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"감정 분석 실패: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"감정 분석 실패: {str(e)}")
