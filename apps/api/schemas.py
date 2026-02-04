from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WorldCreate(BaseModel):
    name: str = "Untitled World"
    snapshot: Dict[str, Any]


class WorldPatch(BaseModel):
    expected_version: int
    event_type: str = "world_update"
    payload: Dict[str, Any]
    snapshot: Dict[str, Any]


class WorldGenerateRequest(BaseModel):
    seed_prompt: str = ""
    style: str = "modern"


class WorldResponse(BaseModel):
    id: str
    name: str
    snapshot: Dict[str, Any]
    version: int


class ChatCreate(BaseModel):
    world_id: str
    user_id: Optional[str] = None


class MessageCreate(BaseModel):
    role: str
    content: str
    desire_score: Optional[int] = None


class AutoChatRequest(BaseModel):
    max_steps: int = 10
    desire_stop: int = 5


class ActorUpdate(BaseModel):
    profile: Dict[str, Any]


class StatsResponse(BaseModel):
    total_messages: int
    total_turns: int
    avg_desire: float
    emotion_distribution: Dict[str, int]
    lexical_diversity: Dict[str, float]
    topic_diversity: Dict[str, float]
    desire_series: List[int]
    avg_turn_duration_sec: float
    trend_summary: str
    trend_direction: str
    trend_delta: float
    trend_window: int


class MemoryConfigUpdate(BaseModel):
    add_mode: Optional[str] = None
    search_mode: Optional[str] = None


class MemoryConfigResponse(BaseModel):
    chat_id: str
    add_mode: str
    search_mode: str
    supported_add_modes: List[str]
    supported_search_modes: List[str]


class SessionConfigUpdate(BaseModel):
    session_time: Optional[str] = None


class SessionConfigResponse(BaseModel):
    chat_id: str
    session_time: str
