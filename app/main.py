from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routers import health, chatbot, auth, chat, admin, diary_view, task
from app.database import engine, Base
from app.models import db_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 이벤트"""
    # 시작 시: 데이터베이스 테이블 생성
    print("🔧 데이터베이스 테이블 초기화 중...")
    Base.metadata.create_all(bind=engine)
    print("✅ 데이터베이스 테이블 준비 완료!\n")

    # 시작 시: PDF 매뉴얼 벡터 스토어 자동 로드
    print("📚 PDF 매뉴얼 벡터 스토어 로딩 중...")
    from app.services.vector_store import get_vector_store_service
    try:
        get_vector_store_service()  # 싱글톤 초기화 트리거
        print("✅ PDF 매뉴얼 벡터 스토어 준비 완료!\n")
    except Exception as e:
        print(f"⚠️  PDF 매뉴얼 로드 실패: {e}")
        print("   /api/chatbot/initialize를 호출하여 수동으로 초기화하세요.\n")

    # 시작 시: 감정 분석 모델 자동 로드
    print("😊 감정 분석 모델 로딩 중...")
    from app.services.emotion_service import get_emotion_service
    try:
        emotion_service = get_emotion_service()  # 싱글톤 초기화 트리거
        if emotion_service.model is not None:
            print(f"✅ 감정 분석 모델 준비 완료! (Device: {emotion_service.device})\n")
        else:
            print("⚠️  감정 분석 모델 로드 실패. 모델 파일을 확인하세요.\n")
    except Exception as e:
        print(f"⚠️  감정 분석 모델 로드 실패: {e}\n")

    # 시작 시: 자동 일기 생성 스케줄러 시작
    print("⏰ 자동 일기 생성 스케줄러 시작 중...")
    from app.services.diary_scheduler import get_diary_scheduler
    scheduler = get_diary_scheduler()
    scheduler.start()
    print()

    yield

    # 종료 시: 스케줄러 중지
    print("\n👋 서버 종료 중...")
    scheduler.stop()


app = FastAPI(
    title="우울감 회복지원 지원 챗봇 API",
    description="우울감을 갖는 분들을 위한 지원 챗봇 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(chatbot.router, prefix="/api/chatbot", tags=["chatbot"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(chat.router)  # prefix와 tags는 router에 이미 포함
app.include_router(admin.router)  # prefix와 tags는 router에 이미 포함
app.include_router(diary_view.router)  # prefix와 tags는 router에 이미 포함
app.include_router(task.router)  # prefix와 tags는 router에 이미 포함

@app.get("/")
async def root():
    return {
        "message": "우울감 회복지원 지원 챗봇 API",
        "version": "1.0.0",
        "docs": "/docs"
    }
