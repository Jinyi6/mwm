from __future__ import annotations

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


def utc_now() -> datetime:
    return datetime.utcnow()


class WorldModel(SQLModel, table=True):
    __tablename__ = "world_models"
    id: str = Field(primary_key=True)
    name: str
    snapshot_json: str
    version: int = Field(default=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class WorldEvent(SQLModel, table=True):
    __tablename__ = "world_events"
    id: str = Field(primary_key=True)
    world_id: str = Field(index=True)
    version: int
    event_type: str
    payload_json: str
    created_at: datetime = Field(default_factory=utc_now)


class Chat(SQLModel, table=True):
    __tablename__ = "chats"
    id: str = Field(primary_key=True)
    world_id: str = Field(index=True)
    user_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now)


class Message(SQLModel, table=True):
    __tablename__ = "messages"
    id: str = Field(primary_key=True)
    chat_id: str = Field(index=True)
    role: str = Field(index=True)
    content: str
    emotion_label: Optional[str] = Field(default=None)
    emotion_confidence: Optional[float] = Field(default=None)
    emotion_method: Optional[str] = Field(default=None)
    desire_score: Optional[int] = Field(default=None)
    topic_tag: Optional[str] = Field(default=None)
    metadata_json: Optional[str] = Field(default=None)
    turn_index: int = Field(index=True)
    created_at: datetime = Field(default_factory=utc_now)


class ActorProfile(SQLModel, table=True):
    __tablename__ = "actor_profiles"
    id: str = Field(primary_key=True)
    chat_id: str = Field(index=True)
    role: str = Field(index=True)
    profile_json: str
    updated_at: datetime = Field(default_factory=utc_now)


class MemoryState(SQLModel, table=True):
    __tablename__ = "memory_states"
    id: str = Field(primary_key=True)
    chat_id: str = Field(index=True)
    role: str = Field(index=True)
    mode: str
    memory_json: str
    updated_at: datetime = Field(default_factory=utc_now)


class MemoryConfig(SQLModel, table=True):
    __tablename__ = "memory_configs"
    chat_id: str = Field(primary_key=True)
    add_mode: str = Field(default="last_60")
    search_mode: str = Field(default="last_60")
    updated_at: datetime = Field(default_factory=utc_now)


class SessionConfig(SQLModel, table=True):
    __tablename__ = "session_configs"
    chat_id: str = Field(primary_key=True)
    session_time: str = Field(default_factory=lambda: utc_now().isoformat())
    updated_at: datetime = Field(default_factory=utc_now)
