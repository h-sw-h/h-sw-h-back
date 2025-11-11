from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "FastAPI Application"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    openai_api_key: Optional[str] = None
    database_url: str  # 필수 환경변수
    redis_url: str = "redis://localhost:6379/0"  # Redis 연결 URL

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings():
    return Settings()
