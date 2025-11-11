# app/repositories/user_repository.py
from sqlalchemy.orm import Session
from app.models.db_models import User
from typing import Optional
import uuid


class UserRepository:
    """사용자 데이터 액세스 레이어"""

    def __init__(self, db: Session):
        self.db = db

    def find_by_email(self, email: str) -> Optional[User]:
        """이메일로 사용자 조회"""
        return self.db.query(User).filter(User.email == email).first()

    def find_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """ID로 사용자 조회"""
        return self.db.query(User).filter(User.user_id == user_id).first()

    def create(self, user: User) -> User:
        """사용자 생성"""
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def update(self, user: User) -> User:
        """사용자 정보 업데이트"""
        self.db.commit()
        self.db.refresh(user)
        return user

    def delete(self, user: User) -> None:
        """사용자 삭제"""
        self.db.delete(user)
        self.db.commit()

    def exists_by_email(self, email: str) -> bool:
        """이메일 존재 여부 확인"""
        return self.db.query(User).filter(User.email == email).count() > 0

    def get_all(self, skip: int = 0, limit: int = 100) -> list[User]:
        """모든 사용자 조회 (페이징)"""
        return self.db.query(User).offset(skip).limit(limit).all()
