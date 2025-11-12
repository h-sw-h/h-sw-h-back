# routers/chat.py
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.chat import (
    SessionCreateRequest,
    SessionCreateResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    SessionEndRequest,
    SessionEndResponse
)
from app.services.chat_session import get_session_manager, ChatSessionManager
from app.services.diary_service import get_diary_service, DiaryService
from app.services.chat_orchestrator import get_chat_orchestrator, ChatOrchestrator
from app.services.vector_store import VectorStoreService, get_vector_store_service
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
