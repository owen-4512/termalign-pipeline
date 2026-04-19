from datetime import datetime

from pydantic import BaseModel


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class ConversationCreate(BaseModel):
    title: str = "新会话"


class ConversationOut(BaseModel):
    id: int
    title: str
    created_at: datetime


class MessageCreate(BaseModel):
    content: str
    language: str = "en"


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    language: str
    created_at: datetime


class ChatResponse(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut
