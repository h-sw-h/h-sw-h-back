# routers/task.py
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.task import (
    TaskGenerateRequest,
    TaskGenerateResponse,
    TaskData
)
from app.schemas.chat import EmotionItem
from app.services.task_generator_service import (
    TaskGeneratorService,
    get_task_generator_service
)
from app.services.chat_session import ChatSessionManager, get_session_manager
from app.routers.auth import get_current_user_id
import traceback

router = APIRouter(prefix="/api/task", tags=["Task"])


@router.post("/generate", response_model=TaskGenerateResponse, summary="과제 생성")
async def generate_task(
    request: TaskGenerateRequest,
    user_id: str = Depends(get_current_user_id),
    task_generator: TaskGeneratorService = Depends(get_task_generator_service),
    session_manager: ChatSessionManager = Depends(get_session_manager)
):
    """
    세션 기반 감정 분석을 통한 과제 생성 API (LangChain 기반)

    **플로우:**
    1. 세션 검증 (user_id 확인)
    2. 세션 대화 내역에서 감정 분석 (상위 5개 emotion + score)
    3. 사용자 질문으로 논문 청크 검색 (PostgreSQL + pgvector)
    4. LangChain LCEL 체인으로 과제(task)와 핵심 효과(core_effect) 생성

    **요청:**
    - session_id: 세션 ID (JWT의 user_id와 일치해야 함)
    - user_query: 사용자 질문
    - k_results: 검색할 논문 청크 개수 (기본값: 2)

    **응답:**
    - task: 생성된 과제 내용
    - core_effect: 핵심 효과 설명
    - emotions: 분석된 5개 감정 리스트
    - sources: 참고 논문 청크 리스트
    """
    try:
        # 1. 세션 검증
        session_info = session_manager.get_session_info(request.session_id)
        
        if not session_info:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        if session_info.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="세션 접근 권한이 없습니다.")
        
        # 2. 세션 기반 과제 생성 (LangChain 체인)
        result = task_generator.generate_task_from_session(
            session_id=request.session_id,
            user_query=request.user_query,
            k_results=request.k_results
        )
        
        # 3. 오류 처리
        if result.get("error"):
            return TaskGenerateResponse(
                success=False,
                message="과제 생성 실패",
                data=None,
                error=result.get("error")
            )
        
        # 4. 감정 데이터 변환
        emotions = []
        for emo in result.get("emotions", []):
            emotions.append(EmotionItem(
                emotion=emo.get("emotion", ""),
                score=emo.get("score", 0.0),
                threshold=emo.get("threshold", 0.5),
                is_active=emo.get("is_active", False)
            ))
        
        # 5. 응답 생성
        task_data = TaskData(
            task=result.get("task", ""),
            core_effect=result.get("core_effect", ""),
            emotions=emotions,
            sources=result.get("sources", [])
        )
        
        return TaskGenerateResponse(
            success=True,
            message="과제가 성공적으로 생성되었습니다.",
            data=task_data,
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"과제 생성 실패: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"과제 생성 실패: {str(e)}")

