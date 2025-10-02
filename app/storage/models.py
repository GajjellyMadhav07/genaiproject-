from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, DateTime, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    user_prompt: Mapped[str] = mapped_column(Text)
    generated_code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    diagram_base64: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    analysis_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)




