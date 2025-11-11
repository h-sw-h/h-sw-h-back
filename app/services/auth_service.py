# app/services/auth_service.py
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from app.models.db_models import User
from app.repositories.user_repository import UserRepository
import uuid

# 설정
SECRET_KEY = "your-secret-key-change-in-production"  # 환경변수로 관리
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30일

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self, db: Session = None):
        self.db = db
        self.user_repository = UserRepository(db)
    
    def hash_password(self, password: str) -> str:
        """비밀번호 해싱 (bcrypt 72바이트 제한 대응)"""
        # bcrypt는 72바이트까지만 처리 가능
        password_bytes = password.encode('utf-8')[:72]
        return pwd_context.hash(password_bytes.decode('utf-8', errors='ignore'))
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """비밀번호 검증 (bcrypt 72바이트 제한 대응)"""
        # 해싱할 때와 동일하게 72바이트로 자르기
        password_bytes = plain_password.encode('utf-8')[:72]
        return pwd_context.verify(password_bytes.decode('utf-8', errors='ignore'), hashed_password)
    
    def create_access_token(self, data: dict) -> str:
        """JWT 토큰 생성"""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """JWT 토큰 검증"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            return None
    
    def register(self, email: str, password: str, nickname: str) -> dict:
        """회원가입"""
        # 이메일 중복 체크
        if self.user_repository.exists_by_email(email):
            raise ValueError("이미 존재하는 이메일입니다")

        # 사용자 생성
        new_user = User(
            user_id=uuid.uuid4(),
            email=email,
            password=self.hash_password(password),
            nickname=nickname,
            created_at=datetime.now(timezone.utc)
        )

        created_user = self.user_repository.create(new_user)

        return {
            "user_id": str(created_user.user_id),
            "email": created_user.email,
            "nickname": created_user.nickname
        }
    
    def login(self, email: str, password: str) -> str:
        """로그인"""
        # 사용자 조회
        user = self.user_repository.find_by_email(email)

        if not user:
            raise ValueError("이메일 또는 비밀번호가 올바르지 않습니다")

        # 비밀번호 검증
        if not self.verify_password(password, user.password):
            raise ValueError("이메일 또는 비밀번호가 올바르지 않습니다")

        # 토큰 생성
        access_token = self.create_access_token(
            data={"user_id": str(user.user_id), "email": user.email}
        )

        return access_token
    
    def get_current_user(self, token: str) -> dict:
        """현재 사용자 정보 가져오기"""
        payload = self.verify_token(token)
        if not payload:
            raise ValueError("유효하지 않은 토큰입니다")

        email = payload.get("email")
        user = self.user_repository.find_by_email(email)

        if not user:
            raise ValueError("사용자를 찾을 수 없습니다")

        return {
            "user_id": str(user.user_id),
            "email": user.email,
            "nickname": user.nickname
        }