# app/models/user.py
from pydantic import BaseModel, EmailStr, Field

class UserCreate(BaseModel):
    email: str = Field(..., example="test@example.com")
    password: str = Field(..., min_length=8, max_length=72, description="비밀번호 (8-72자)", example="password123")
    nickname: str = Field(..., min_length=2, max_length=50, description="닉네임 (2-50자)", example="테스트유저")

class UserLogin(BaseModel):
    email: str = Field(..., example="test@example.com")
    password: str = Field(..., max_length=72, example="password123")

class Token(BaseModel):
    access_token: str = Field(..., example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    token_type: str = Field(..., example="bearer")