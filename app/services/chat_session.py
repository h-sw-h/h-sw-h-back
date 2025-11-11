# services/chat_session.py
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import uuid
import redis
from app.config import get_settings

class ChatSessionManager:
    """
    대화 세션 관리 (Redis 기반)
    - TTL 30분 자동 만료
    - 메시지 히스토리 저장
    - 세션 메타데이터 관리
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Args:
            redis_client: Redis 클라이언트 (선택). 없으면 자동 생성
        """
        if redis_client:
            self.redis = redis_client
        else:
            settings = get_settings()
            self.redis = redis.from_url(
                settings.redis_url,
                decode_responses=True  # 자동 문자열 디코딩
            )

        self.session_ttl = 30 * 60  # 30분 (초 단위)

    def create_session(self, user_id: str) -> str:
        """
        새 세션 생성

        Returns:
            session_id: 생성된 세션 ID
        """
        session_id = user_id
        session_key = f"session:{session_id}"

        session_data = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "message_count": 0
        }

        # 세션 메타데이터 저장 (Hash)
        self.redis.hset(session_key, mapping=session_data)
        self.redis.expire(session_key, self.session_ttl)

        # 메시지 리스트 초기화 (List)
        messages_key = f"messages:{session_id}"
        self.redis.delete(messages_key)  # 혹시 모를 기존 데이터 삭제
        self.redis.expire(messages_key, self.session_ttl)

        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """
        세션에 메시지 추가

        Args:
            session_id: 세션 ID
            role: "user" 또는 "assistant"
            content: 메시지 내용

        Returns:
            성공 여부
        """
        session_key = f"session:{session_id}"

        # 세션 존재 확인
        if not self.redis.exists(session_key):
            return False

        # 메시지 객체 생성
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        # 메시지 리스트에 추가 (RPUSH: 오른쪽에 추가)
        messages_key = f"messages:{session_id}"
        self.redis.rpush(messages_key, json.dumps(message))

        # 세션 메타데이터 업데이트
        self.redis.hset(session_key, "last_activity", datetime.now().isoformat())
        self.redis.hincrby(session_key, "message_count", 1)

        # TTL 갱신 (활동 시 30분 연장)
        self.redis.expire(session_key, self.session_ttl)
        self.redis.expire(messages_key, self.session_ttl)

        return True

    def get_messages(self, session_id: str, limit: int = 10) -> List[Dict]:
        """
        최근 N개 메시지 가져오기 (컨텍스트 윈도우용)

        Args:
            session_id: 세션 ID
            limit: 가져올 메시지 개수 (기본 10개)

        Returns:
            메시지 리스트
        """
        messages_key = f"messages:{session_id}"

        # 최근 limit개 가져오기 (음수 인덱스 사용)
        raw_messages = self.redis.lrange(messages_key, -limit, -1)

        messages = []
        for msg_str in raw_messages:
            try:
                messages.append(json.loads(msg_str))
            except json.JSONDecodeError:
                continue

        return messages

    def get_full_conversation(self, session_id: str) -> List[Dict]:
        """
        전체 대화 가져오기 (요약 및 일기 생성용)

        Returns:
            전체 메시지 리스트
        """
        messages_key = f"messages:{session_id}"

        # 전체 메시지 가져오기
        raw_messages = self.redis.lrange(messages_key, 0, -1)

        messages = []
        for msg_str in raw_messages:
            try:
                messages.append(json.loads(msg_str))
            except json.JSONDecodeError:
                continue

        return messages

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        세션 메타데이터 조회

        Returns:
            세션 정보 또는 None
        """
        session_key = f"session:{session_id}"

        if not self.redis.exists(session_key):
            return None

        return self.redis.hgetall(session_key)

    def extend_session(self, session_id: str, minutes: int = 30):
        """
        세션 TTL 연장

        Args:
            session_id: 세션 ID
            minutes: 연장할 시간 (분)
        """
        ttl = minutes * 60
        self.redis.expire(f"session:{session_id}", ttl)
        self.redis.expire(f"messages:{session_id}", ttl)

    # def delete_session(self, session_id: str):
    #     """
    #     세션 즉시 삭제 (수동 종료 시)
    #     """
    #     self.redis.delete(f"session:{session_id}")
    #     self.redis.delete(f"messages:{session_id}")

    def session_exists(self, session_id: str) -> bool:
        """
        세션 존재 확인
        """
        return self.redis.exists(f"session:{session_id}") > 0


# 전역 세션 관리자 인스턴스 (싱글톤)
_session_manager: Optional[ChatSessionManager] = None

def get_session_manager() -> ChatSessionManager:
    """
    세션 관리자 의존성 주입
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = ChatSessionManager()
    return _session_manager
