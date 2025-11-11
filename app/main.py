from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routers import health, chatbot, auth, chat
from app.database import engine, Base
from app.models import db_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 이벤트"""
    # 시작 시: 데이터베이스 테이블 생성
    print("🔧 데이터베이스 테이블 초기화 중...")
    Base.metadata.create_all(bind=engine)
    print("✅ 데이터베이스 테이블 준비 완료!")

    yield

    # 종료 시: 정리 작업 (필요시)
    print("👋 서버 종료 중...")


app = FastAPI(
    title="은둔/고립 청년 사회복귀 지원 챗봇 API",
    description="은둔/고립 청년의 원활한 사회복귀를 돕는 RAG 기반 챗봇 API",
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

@app.get("/")
async def root():
    return {
        "message": "은둔/고립 청년 사회복귀 지원 챗봇 API",
        "version": "1.0.0",
        "docs": "/docs"
    }
