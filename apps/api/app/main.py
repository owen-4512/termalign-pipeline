from datetime import datetime
from io import StringIO
import csv

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .config import settings
from .database import Base, SessionLocal, engine, get_db
from .models import Conversation, Message, User
from .openai_client import llm_client
from .schemas import (
    ChatResponse,
    ConversationCreate,
    ConversationOut,
    LoginRequest,
    MessageCreate,
    MessageOut,
    TokenResponse,
)
from .security import create_access_token, decode_token, hash_password, verify_password

app = FastAPI(title="TermAlign API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.replace("Bearer ", "")
    try:
        payload = decode_token(token)
        user_id = int(payload["sub"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user


@app.on_event("startup")
def startup():
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        admin = db.scalar(select(User).where(User.email == "admin@example.com"))
        if not admin:
            db.add(
                User(
                    email="admin@example.com",
                    password_hash=hash_password("admin123"),
                    role="admin",
                    name="管理员",
                )
            )
        student = db.scalar(select(User).where(User.email == "student@example.com"))
        if not student:
            db.add(
                User(
                    email="student@example.com",
                    password_hash=hash_password("student123"),
                    role="student",
                    name="示例学生",
                )
            )
        db.commit()


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(user.id, user.role), role=user.role)


@app.post("/chat/conversations", response_model=ConversationOut)
def create_conversation(
    payload: ConversationCreate,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    conv = Conversation(title=payload.title, user_id=user.id)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


@app.get("/chat/conversations", response_model=list[ConversationOut])
def list_conversations(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(
        select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.created_at.desc())
    ).all()
    return rows


@app.get("/chat/conversations/{conversation_id}/messages", response_model=list[MessageOut])
def list_messages(conversation_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    conv = db.get(Conversation, conversation_id)
    if not conv or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    rows = db.scalars(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.asc())
    ).all()
    return rows


@app.post("/chat/conversations/{conversation_id}/messages", response_model=ChatResponse)
def send_message(
    conversation_id: int,
    payload: MessageCreate,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    conv = db.get(Conversation, conversation_id)
    if not conv or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=payload.content,
        language=payload.language,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    answer, token_used = llm_client.generate_reply(payload.content, payload.language)
    assistant_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        language=payload.language,
        tokens_used=token_used,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return ChatResponse(user_message=user_msg, assistant_message=assistant_msg)


@app.get("/admin/messages")
def admin_messages(
    user_email: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    stmt = (
        select(Message, User.email, Conversation.title)
        .join(Conversation, Conversation.id == Message.conversation_id)
        .join(User, User.id == Conversation.user_id)
        .order_by(Message.created_at.desc())
    )
    if user_email:
        stmt = stmt.where(User.email.ilike(f"%{user_email}%"))
    if keyword:
        stmt = stmt.where(or_(Message.content.ilike(f"%{keyword}%"), Conversation.title.ilike(f"%{keyword}%")))
    if start:
        stmt = stmt.where(Message.created_at >= datetime.fromisoformat(start))
    if end:
        stmt = stmt.where(Message.created_at <= datetime.fromisoformat(end))

    rows = db.execute(stmt.limit(500)).all()
    return [
        {
            "id": m.id,
            "conversation_id": m.conversation_id,
            "conversation_title": title,
            "user_email": email,
            "role": m.role,
            "content": m.content,
            "language": m.language,
            "created_at": m.created_at,
        }
        for m, email, title in rows
    ]


@app.get("/admin/messages/export")
def admin_export(
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        select(Message, User.email, Conversation.title)
        .join(Conversation, Conversation.id == Message.conversation_id)
        .join(User, User.id == Conversation.user_id)
        .order_by(Message.created_at.desc())
        .limit(2000)
    ).all()

    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["message_id", "conversation_id", "conversation_title", "user_email", "role", "content", "created_at"])
    for m, email, title in rows:
        writer.writerow([m.id, m.conversation_id, title, email, m.role, m.content, m.created_at.isoformat()])

    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=messages.csv"},
    )


@app.get("/admin/stats")
def admin_stats(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    message_count = db.scalar(select(func.count(Message.id))) or 0
    conversation_count = db.scalar(select(func.count(Conversation.id))) or 0
    active_users = db.scalar(select(func.count(func.distinct(Conversation.user_id)))) or 0
    return {
        "message_count": message_count,
        "conversation_count": conversation_count,
        "active_users": active_users,
    }
