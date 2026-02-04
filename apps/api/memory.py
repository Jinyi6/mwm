from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from sqlmodel import Session, select

from .models import Message, MemoryState, MemoryConfig
from .utils import dumps_json, loads_json

EMOTION_LABELS = [
    "joy",
    "sad",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "neutral",
    "other",
]

EMOTION_KEYWORDS = {
    "joy": ["开心", "高兴", "快乐", "满意", "喜欢", "太好了"],
    "sad": ["难过", "伤心", "失落", "沮丧"],
    "anger": ["生气", "愤怒", "恼火", "烦"],
    "fear": ["害怕", "恐惧", "担心", "紧张"],
    "surprise": ["惊讶", "意外", "没想到"],
    "disgust": ["恶心", "反感", "讨厌"],
}

TOPIC_KEYWORDS = {
    "关系": ["你", "我", "我们", "关系"],
    "目标": ["计划", "目标", "安排"],
    "冲突": ["争执", "矛盾", "冲突"],
    "日常": ["吃", "喝", "走", "逛", "天气"],
}

DEFAULT_ADD_MODE = "last_60"
DEFAULT_SEARCH_MODE = "last_60"

ADD_MODE_LIMITS = {
    "last_20": 20,
    "last_60": 60,
    "last_100": 100,
}

SEARCH_MODE_LIMITS = {
    "last_20": 20,
    "last_60": 60,
    "last_100": 100,
}


def supported_add_modes() -> List[str]:
    return list(ADD_MODE_LIMITS.keys()) + ["none"]


def supported_search_modes() -> List[str]:
    return ["state", "none"] + list(SEARCH_MODE_LIMITS.keys())


def get_memory_config(session: Session, chat_id: str) -> Dict[str, str]:
    config = session.get(MemoryConfig, chat_id)
    if not config:
        return {"add_mode": DEFAULT_ADD_MODE, "search_mode": DEFAULT_SEARCH_MODE}
    return {"add_mode": config.add_mode, "search_mode": config.search_mode}


def set_memory_config(session: Session, chat_id: str, add_mode: Optional[str], search_mode: Optional[str]) -> Dict[str, str]:
    current = get_memory_config(session, chat_id)
    add = add_mode or current["add_mode"]
    search = search_mode or current["search_mode"]
    config = session.get(MemoryConfig, chat_id)
    if not config:
        config = MemoryConfig(chat_id=chat_id, add_mode=add, search_mode=search)
        session.add(config)
    else:
        config.add_mode = add
        config.search_mode = search
    return {"add_mode": add, "search_mode": search}


def rule_emotion(text: str) -> Tuple[str, float]:
    text = text or ""
    scores = {label: 0 for label in EMOTION_LABELS}
    for label, words in EMOTION_KEYWORDS.items():
        for w in words:
            if w in text:
                scores[label] += 1
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "neutral", 0.4
    confidence = min(0.95, 0.5 + 0.15 * scores[best])
    return best, confidence


def rule_topic(text: str) -> str:
    text = text or ""
    for topic, words in TOPIC_KEYWORDS.items():
        for w in words:
            if w in text:
                return topic
    return "misc"


def _recent_messages(session: Session, chat_id: str, limit: int) -> List[Message]:
    stmt = (
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    items = session.exec(stmt).all()
    return list(reversed(items))


def update_memory(session: Session, chat_id: str, role: str, mode: str = DEFAULT_ADD_MODE) -> None:
    if mode == "none":
        memory_items: List[Dict[str, Any]] = []
    else:
        limit = ADD_MODE_LIMITS.get(mode, ADD_MODE_LIMITS[DEFAULT_ADD_MODE])
        items = _recent_messages(session, chat_id, limit)
        memory_items = [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in items
        ]
    state = session.get(MemoryState, f"{chat_id}:{role}")
    payload = dumps_json({"mode": mode, "items": memory_items})
    if state:
        state.memory_json = payload
    else:
        state = MemoryState(
            id=f"{chat_id}:{role}",
            chat_id=chat_id,
            role=role,
            mode=mode,
            memory_json=payload,
        )
        session.add(state)


def get_memory(session: Session, chat_id: str, role: str) -> List[Dict[str, Any]]:
    state = session.get(MemoryState, f"{chat_id}:{role}")
    if not state:
        return []
    data = loads_json(state.memory_json, {"items": []})
    return data.get("items", [])


def search_memory(session: Session, chat_id: str, role: str, mode: str = DEFAULT_SEARCH_MODE) -> List[Dict[str, Any]]:
    if mode == "none":
        return []
    if mode == "state":
        return get_memory(session, chat_id, role)
    limit = SEARCH_MODE_LIMITS.get(mode)
    if not limit:
        limit = SEARCH_MODE_LIMITS[DEFAULT_SEARCH_MODE]
    items = _recent_messages(session, chat_id, limit)
    return [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in items]
