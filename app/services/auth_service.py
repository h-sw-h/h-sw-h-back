# app/services/auth_service.py
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
import json
import os

# 설정
SECRET_KEY = "your-secret-key-change-in-production"  # 환경변수로 관리
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30일

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def __init__(self):
        self.users_file = "data/users.json"
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.users_file):
            self._save_users({})
    
    def _load_users(self) -> dict:
        """사용자 데이터 로드"""
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def _save_users(self, users: dict):
        """사용자 데이터 저장"""
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    
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
        users = self._load_users()
        
        # 이메일 중복 체크
        if email in users:
            raise ValueError("이미 존재하는 이메일입니다")
        
        # 사용자 생성
        import uuid
        user_id = str(uuid.uuid4())
        users[email] = {
            "user_id": user_id,
            "email": email,
            "password": self.hash_password(password),
            "nickname": nickname,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        self._save_users(users)
        return {"user_id": user_id, "email": email, "nickname": nickname}
    
    def login(self, email: str, password: str) -> str:
        """로그인"""
        users = self._load_users()
        
        # 사용자 존재 확인
        if email not in users:
            raise ValueError("이메일 또는 비밀번호가 올바르지 않습니다")
        
        user = users[email]
        
        # 비밀번호 검증
        if not self.verify_password(password, user["password"]):
            raise ValueError("이메일 또는 비밀번호가 올바르지 않습니다")
        
        # 토큰 생성
        access_token = self.create_access_token(
            data={"user_id": user["user_id"], "email": email}
        )
        
        return access_token
    
    def get_current_user(self, token: str) -> dict:
        """현재 사용자 정보 가져오기"""
        payload = self.verify_token(token)
        if not payload:
            raise ValueError("유효하지 않은 토큰입니다")
        
        users = self._load_users()
        email = payload.get("email")
        
        if email not in users:
            raise ValueError("사용자를 찾을 수 없습니다")
        
        user = users[email]
        return {
            "user_id": user["user_id"],
            "email": user["email"],
            "nickname": user["nickname"]
        }