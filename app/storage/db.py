import json
import os
from contextlib import contextmanager
from typing import Iterable, List, Dict, Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import settings
from app.storage.models import Base, ChatMessage


def get_engine():
    os.makedirs(os.path.dirname(settings.database_path), exist_ok=True)
    url = f"sqlite:///{settings.database_path}"
    return create_engine(url, connect_args={"check_same_thread": False})


engine = get_engine()
Base.metadata.create_all(engine)


@contextmanager
def session_scope():
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_message(
    session_id: str,
    user_prompt: str,
    generated_code: str | None,
    diagram_base64: str | None,
    analysis: Dict[str, Any] | None,
) -> int:
    with session_scope() as s:
        msg = ChatMessage(
            session_id=session_id,
            user_prompt=user_prompt,
            generated_code=generated_code,
            diagram_base64=diagram_base64,
            analysis_json=json.dumps(analysis) if analysis else None,
        )
        s.add(msg)
        s.flush()
        return msg.id


def fetch_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    with session_scope() as s:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        rows: Iterable[ChatMessage] = s.execute(stmt).scalars().all()
        history: List[Dict[str, Any]] = []
        for r in rows:
            history.append(
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "user_prompt": r.user_prompt,
                    "generated_code": r.generated_code,
                    "diagram_base64": r.diagram_base64,
                    "analysis_json": r.analysis_json,
                    "created_at": r.created_at.isoformat(),
                }
            )
        return list(reversed(history))




