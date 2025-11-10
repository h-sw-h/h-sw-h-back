from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import health, chatbot

app = FastAPI(
    title="은둔/고립 청년 사회복귀 지원 챗봇 API",
    description="은둔/고립 청년의 원활한 사회복귀를 돕는 RAG 기반 챗봇 API",
    version="1.0.0"
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

@app.get("/")
async def root():
    return {
        "message": "은둔/고립 청년 사회복귀 지원 챗봇 API",
        "version": "1.0.0",
        "docs": "/docs"
    }
