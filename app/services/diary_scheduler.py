# services/diary_scheduler.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timezone, timedelta
from typing import List
import traceback

from app.services.chat_session import get_session_manager
from app.services.chat_orchestrator import ChatOrchestrator
from app.services.diary_service import get_diary_service
from app.services.vector_store import get_vector_store_service


class DiaryScheduler:
    """
    자동 일기 생성 스케줄러

    - 매일 새벽 6시에 실행
    - Redis의 모든 세션을 일기로 변환
    - PostgreSQL에 저장 후 세션 삭제
    """

    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
        self.session_manager = get_session_manager()
        self.diary_service = get_diary_service()

    def start(self):
        """스케줄러 시작"""
        # 매일 새벽 6시 실행
        self.scheduler.add_job(
            self.auto_create_diaries,
            trigger=CronTrigger(hour=6, minute=0),  # 06:00 KST
            id="auto_diary_creation",
            name="자동 일기 생성",
            replace_existing=True
        )

        self.scheduler.start()
        print("✅ 일기 자동 생성 스케줄러 시작 (매일 06:00)")

    def stop(self):
        """스케줄러 중지"""
        self.scheduler.shutdown()
        print("👋 일기 자동 생성 스케줄러 중지")

    async def auto_create_diaries(self):
        """
        자동 일기 생성 작업

        1. Redis에서 모든 세션 찾기
        2. 각 세션을 일기로 변환
        3. PostgreSQL에 저장
        4. 세션 삭제
        """
        try:
            print(f"\n{'='*60}")
            print(f"🌅 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 자동 일기 생성 시작")
            print(f"{'='*60}\n")

            # 1. 모든 세션 키 가져오기
            session_keys = self.session_manager.redis.keys("session:*")

            if not session_keys:
                print("📭 처리할 세션이 없습니다.")
                return

            print(f"📦 총 {len(session_keys)}개 세션 발견\n")

            success_count = 0
            fail_count = 0

            # 2. 각 세션 처리
            for session_key in session_keys:
                session_id = session_key.replace("session:", "")

                try:
                    # 세션 정보 가져오기
                    session_info = self.session_manager.get_session_info(session_id)

                    if not session_info:
                        print(f"⚠️  세션 {session_id[:8]}... - 세션 정보 없음 (스킵)")
                        continue

                    user_id = session_info.get("user_id")
                    message_count = int(session_info.get("message_count", 0))

                    # 메시지가 없으면 스킵
                    if message_count == 0:
                        print(f"⚠️  세션 {session_id[:8]}... (user: {user_id[:8]}...) - 메시지 없음 (삭제)")
                        self.session_manager.redis.delete(session_key)
                        self.session_manager.redis.delete(f"messages:{session_id}")
                        continue

                    # 일기 생성
                    print(f"📝 세션 {session_id[:8]}... (user: {user_id[:8]}..., {message_count}개 메시지)")

                    diary_result = await self._summarize_to_diary(session_id)

                    if not diary_result or not diary_result.get("diary_text"):
                        print(f"   ❌ 일기 생성 실패 (요약 실패)")
                        fail_count += 1
                        continue

                    # 일기 저장 (diary_text만 사용)
                    diary_id = self.diary_service.save_diary(
                        user_id=user_id,
                        diary_content=diary_result["diary_text"],  # 문자열 추출
                        alternative_perspective=diary_result.get("alternative_perspective", ""),
                        message_count=session_info.get("message_count", 0)
                    )

                    print(f"   ✅ 일기 저장 완료 (diary_id: {diary_id[:8]}...)")

                    # 세션 삭제
                    self.session_manager.redis.delete(session_key)
                    self.session_manager.redis.delete(f"messages:{session_id}")

                    success_count += 1

                except Exception as e:
                    print(f"   ❌ 처리 실패: {str(e)}")
                    print(f"   {traceback.format_exc()}")
                    fail_count += 1

            # 3. 결과 출력
            print(f"\n{'='*60}")
            print(f"📊 처리 결과:")
            print(f"   - 성공: {success_count}개")
            print(f"   - 실패: {fail_count}개")
            print(f"   - 총: {len(session_keys)}개")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"❌ 자동 일기 생성 작업 실패: {e}")
            print(traceback.format_exc())

    async def _summarize_to_diary(self, session_id: str) -> dict:
        """
        세션을 일기로 요약

        Args:
            session_id: 세션 ID

        Returns:
            일기 딕셔너리 {"diary_text": str, "alternative_perspective": str}
        """
        try:
            # Orchestrator 생성
            vector_store = get_vector_store_service()
            orchestrator = ChatOrchestrator(
                session_manager=self.session_manager,
                diary_service=self.diary_service,
                vector_store=vector_store
            )

            # 요약 (딕셔너리 반환)
            diary_result = orchestrator.summarize_conversation_to_diary(session_id)

            return diary_result

        except Exception as e:
            print(f"      요약 실패: {e}")
            return {"diary_text": "", "alternative_perspective": ""}


# ============================================
# 싱글톤 패턴
# ============================================
_scheduler_instance: DiaryScheduler = None

def get_diary_scheduler() -> DiaryScheduler:
    """
    일기 스케줄러 인스턴스 가져오기 (싱글톤)
    """
    global _scheduler_instance

    if _scheduler_instance is None:
        _scheduler_instance = DiaryScheduler()

    return _scheduler_instance
