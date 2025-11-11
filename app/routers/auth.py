# app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.models.user import UserCreate, UserLogin, Token
from app.services.auth_service import AuthService
from app.database import get_db

router = APIRouter()
security = HTTPBearer()

@router.post("/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """회원가입"""
    try:
        auth_service = AuthService(db)
        result = auth_service.register(
            email=user.email,
            password=user.password,
            nickname=user.nickname
        )
        return {
            "status": "success",
            "message": "회원가입이 완료되었습니다",
            "data": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    """로그인"""
    try:
        auth_service = AuthService(db)
        token = auth_service.login(
            email=user.email,
            password=user.password
        )
        return {
            "access_token": token,
            "token_type": "bearer"
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

@router.get("/me")
async def get_me(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """현재 사용자 정보"""
    try:
        auth_service = AuthService(db)
        user = auth_service.get_current_user(credentials.credentials)
        return {"status": "success", "data": user}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

# Dependency: 인증 필요한 엔드포인트에서 사용
async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> str:
    """현재 사용자 ID 가져오기 (Dependency)"""
    try:
        auth_service = AuthService(db)
        user = auth_service.get_current_user(credentials.credentials)
        return user["user_id"]
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))