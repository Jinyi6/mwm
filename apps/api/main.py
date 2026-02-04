from __future__ import annotations

from typing import Any, Dict, List
import re
import json
import logging
import time
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlmodel import select

from .db import init_db, get_session
from .models import WorldModel, WorldEvent, Chat, Message, ActorProfile, MemoryState, MemoryConfig, SessionConfig
from .schemas import (
    WorldCreate,
    WorldPatch,
    WorldGenerateRequest,
    WorldResponse,
    ChatCreate,
    MessageCreate,
    AutoChatRequest,
    ActorUpdate,
    StatsResponse,
    MemoryConfigUpdate,
    MemoryConfigResponse,
    SessionConfigUpdate,
    SessionConfigResponse,
    NpcGenerateRequest,
    NpcGenerateResponse,
)
from .utils import new_id, new_session_id, dumps_json, loads_json, clamp_int
from .llm import (
    generate_world,
    generate_npc_profile,
    generate_npc_reply,
    generate_npc_reply_with_sp,
    generate_npc_reply_stream,
    generate_npc_reply_stream_with_sp,
    generate_user_reply,
    classify_emotion_llm,
    tag_topic_llm,
)
from .memory import (
    rule_emotion,
    rule_topic,
    update_memory,
    search_memory,
    get_memory,
    DEFAULT_ADD_MODE,
    get_memory_config,
    set_memory_config,
    supported_add_modes,
    supported_search_modes,
)
from .stats import compute_stats

app = FastAPI(title="AI Town Lite")

logger = logging.getLogger("mwm.api")

WORLD_CACHE: Dict[tuple, Dict[str, Any]] = {}
HISTORY_CACHE: Dict[tuple, tuple[float, List[Dict[str, Any]]]] = {}
STATS_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_SEC = 2.0


def _world_snapshot_cached(world: WorldModel) -> Dict[str, Any]:
    key = (world.id, world.version)
    cached = WORLD_CACHE.get(key)
    if cached is not None:
        return cached
    snapshot = loads_json(world.snapshot_json, {})
    WORLD_CACHE[key] = snapshot
    return snapshot


def _history_cached(session, chat_id: str, role: str, search_mode: str) -> List[Dict[str, Any]]:
    key = (chat_id, role, search_mode)
    now = time.time()
    cached = HISTORY_CACHE.get(key)
    if cached and now - cached[0] < CACHE_TTL_SEC:
        return cached[1]
    items = _recent_history(session, chat_id, role, search_mode)
    HISTORY_CACHE[key] = (now, items)
    return items


def _stats_cached(session, chat_id: str) -> Dict[str, Any]:
    now = time.time()
    cached = STATS_CACHE.get(chat_id)
    if cached and now - cached[0] < CACHE_TTL_SEC:
        return cached[1]
    stats = compute_stats(session, chat_id)
    STATS_CACHE[chat_id] = (now, stats)
    return stats


def _invalidate_chat_cache(chat_id: str) -> None:
    STATS_CACHE.pop(chat_id, None)
    for key in list(HISTORY_CACHE.keys()):
        if key[0] == chat_id:
            HISTORY_CACHE.pop(key, None)


def _invalidate_world_cache(world_id: str) -> None:
    for key in list(WORLD_CACHE.keys()):
        if key[0] == world_id:
            WORLD_CACHE.pop(key, None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code, duration)
    return response


@app.on_event("startup")
def _startup() -> None:
    init_db()


def _default_world() -> Dict[str, Any]:
    return {
        "scene": {
            "setting": "一座带有电影质感的当代城市，故事在雨后的街道展开。",
            "ontology": {"user_profile": {}, "user_state_machine": {}},
        },
        "event_schema": {"plot_hooks": ["一次未完成的约定", "一条迟到的消息"]},
        "norms": ["保持角色一致性", "对话风格现代、克制"],
        "beliefs": [],
        "memory": {"semantic": [], "episodic": [], "working": []},
    }


def _list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_world(snapshot: Any) -> Dict[str, Any]:
    base = _default_world()
    data = _dict(snapshot)

    scene = _dict(data.get("scene"))
    ontology = _dict(scene.get("ontology"))
    event_schema = _dict(data.get("event_schema"))
    memory = _dict(data.get("memory"))

    return {
        "scene": {
            "setting": scene.get("setting") if isinstance(scene.get("setting"), str) else base["scene"]["setting"],
            "ontology": {
                "user_profile": _dict(ontology.get("user_profile")),
                "user_state_machine": _dict(ontology.get("user_state_machine")),
            },
        },
        "event_schema": {"plot_hooks": _list(event_schema.get("plot_hooks")) or base["event_schema"]["plot_hooks"]},
        "norms": _list(data.get("norms")) or base["norms"],
        "beliefs": _list(data.get("beliefs")),
        "memory": {
            "semantic": _list(memory.get("semantic")),
            "episodic": _list(memory.get("episodic")),
            "working": _list(memory.get("working")),
        },
    }


@app.get("/config")
def get_config() -> Dict[str, Any]:
    return {
        "emotion_labels": ["joy", "sad", "anger", "fear", "surprise", "disgust", "neutral", "other"],
        "default_desire": 6,
        "memory": {
            "add_mode": "last_60",
            "search_mode": "last_60",
            "supported_add_modes": supported_add_modes(),
            "supported_search_modes": supported_search_modes(),
        },
    }


@app.post("/worlds/generate", response_model=WorldResponse)
def world_generate(payload: WorldGenerateRequest) -> WorldResponse:
    snapshot = _normalize_world(generate_world(payload.seed_prompt, payload.style))
    world = WorldModel(
        id=new_id(),
        name="Generated World",
        snapshot_json=dumps_json(snapshot),
        version=1,
    )
    with get_session() as session:
        session.add(world)
        session.commit()
        session.refresh(world)
        return WorldResponse(id=world.id, name=world.name, snapshot=snapshot, version=world.version)


@app.post("/npc/generate", response_model=NpcGenerateResponse)
def npc_generate(payload: NpcGenerateRequest) -> NpcGenerateResponse:
    profile = generate_npc_profile(payload.world, payload.style)
    return NpcGenerateResponse(profile=profile)


@app.post("/worlds", response_model=WorldResponse)
def world_create(payload: WorldCreate) -> WorldResponse:
    snapshot = _normalize_world(payload.snapshot)
    world = WorldModel(
        id=new_id(),
        name=payload.name,
        snapshot_json=dumps_json(snapshot),
        version=1,
    )
    with get_session() as session:
        session.add(world)
        session.commit()
        session.refresh(world)
        return WorldResponse(id=world.id, name=world.name, snapshot=snapshot, version=world.version)


@app.get("/worlds/{world_id}", response_model=WorldResponse)
def world_get(world_id: str) -> WorldResponse:
    with get_session() as session:
        world = session.get(WorldModel, world_id)
        if not world:
            raise HTTPException(status_code=404, detail="world not found")
        snapshot = _normalize_world(_world_snapshot_cached(world))
        return WorldResponse(id=world.id, name=world.name, snapshot=snapshot, version=world.version)


@app.get("/worlds/{world_id}/events")
def world_events(world_id: str) -> List[Dict[str, Any]]:
    with get_session() as session:
        items = session.exec(select(WorldEvent).where(WorldEvent.world_id == world_id).order_by(WorldEvent.version)).all()
        return [
            {
                "id": e.id,
                "version": e.version,
                "event_type": e.event_type,
                "payload": loads_json(e.payload_json, {}),
                "created_at": e.created_at.isoformat(),
            }
            for e in items
        ]


@app.patch("/worlds/{world_id}", response_model=WorldResponse)
def world_patch(world_id: str, payload: WorldPatch) -> WorldResponse:
    with get_session() as session:
        world = session.get(WorldModel, world_id)
        if not world:
            raise HTTPException(status_code=404, detail="world not found")
        if world.version != payload.expected_version:
            raise HTTPException(status_code=409, detail="version mismatch")
        snapshot = _normalize_world(payload.snapshot)
        new_version = world.version + 1
        event = WorldEvent(
            id=new_id(),
            world_id=world_id,
            version=new_version,
            event_type=payload.event_type,
            payload_json=dumps_json(payload.payload),
        )
        session.add(event)
        world.snapshot_json = dumps_json(snapshot)
        _invalidate_world_cache(world.id)
        world.version = new_version
        session.commit()
        return WorldResponse(id=world.id, name=world.name, snapshot=snapshot, version=world.version)


@app.post("/chats", response_model=Dict[str, Any])
def chat_create(payload: ChatCreate) -> Dict[str, Any]:
    with get_session() as session:
        world = session.get(WorldModel, payload.world_id)
        if not world:
            raise HTTPException(status_code=404, detail="world not found")
        chat_id = new_session_id()
        for _ in range(5):
            if session.get(Chat, chat_id) is None:
                break
            chat_id = new_session_id()
        chat = Chat(id=chat_id, world_id=payload.world_id, user_id=payload.user_id)
        session.add(chat)
        session.add(SessionConfig(chat_id=chat_id))
        session.add(MemoryConfig(chat_id=chat_id))
        session.commit()
        session.refresh(chat)
        return {"id": chat.id, "world_id": chat.world_id}


@app.get("/chats/{chat_id}", response_model=Dict[str, Any])
def chat_get(chat_id: str) -> Dict[str, Any]:
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        msgs = session.exec(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.turn_index, Message.created_at)
        ).all()
        return {
            "id": chat.id,
            "world_id": chat.world_id,
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "emotion_label": m.emotion_label,
                    "emotion_confidence": m.emotion_confidence,
                    "emotion_method": m.emotion_method,
                    "desire_score": m.desire_score,
                    "topic_tag": m.topic_tag,
                    "turn_index": m.turn_index,
                    "created_at": m.created_at.isoformat(),
                }
                for m in msgs
            ],
        }


def _build_context(world: Dict[str, Any], history: List[Dict[str, Any]], actor_profile: Dict[str, Any]) -> str:
    return (
        "World:\n" + dumps_json(world) +
        "\nProfile:\n" + dumps_json(actor_profile) +
        "\nHistory:\n" + dumps_json(history)
    )


def _build_user_context(history: List[Dict[str, Any]], actor_profile: Dict[str, Any]) -> str:
    return (
        "Profile:\n" + dumps_json(actor_profile) +
        "\nHistory:\n" + dumps_json(history)
    )


def _recent_context(session, chat_id: str, limit: int = 60) -> List[Dict[str, Any]]:
    stmt = (
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    items = session.exec(stmt).all()
    items = list(reversed(items))
    return [{"role": m.role, "content": m.content} for m in items]


def _volc_sp_from_profile(world: Dict[str, Any], npc_profile: Dict[str, Any]) -> str:
    role_name = npc_profile.get("role_name", "NPC")
    user_name = npc_profile.get("user_name", "用户")
    init_role_sp = npc_profile.get("init_role_sp", "你是一个有自己背景与动机的角色。")
    user_info = npc_profile.get("user_info", "你和用户正在进行日常对话。")
    golden_sp = npc_profile.get("golden_sp", "")
    defaults = [
        "你使用口语进行表达，必要时可用括号描述动作和情绪。",
        "你需要尽可能引导用户跟你进行交流，你不应该表现地太AI。",
        "每次回复推进剧情，提出问题或下一步行动，抛出具体线索或选择。",
        "避免重复上一轮NPC内容或问句；若用户已回应，请确认并给出新进展。",
        "避免使用“你怎么想/你怎么看/你觉得呢”等泛问句，改用具体行动或选择。",
        "仅使用中文回答。",
    ]
    if not golden_sp:
        golden_sp = "\n".join(defaults)
    else:
        for line in defaults:
            if line not in golden_sp:
                golden_sp = f"{golden_sp}\n{line}"
    return (
        f"{init_role_sp}\n"
        f"{user_info}\n"
        f"{golden_sp}\n"
        f"现在请扮演{role_name}，{role_name}正在和{user_name}对话。\n"
        f"World: {dumps_json(world)}"
    )


def _recent_history(session, chat_id: str, role: str, search_mode: str) -> List[Dict[str, Any]]:
    items = search_memory(session, chat_id, role, mode=search_mode)
    return [{"role": m["role"], "content": m["content"]} for m in items]


def _next_turn_index(last: Message | None, role: str) -> int:
    if not last:
        return 1
    if last.role != role:
        return last.turn_index
    return last.turn_index + 1


def _actor_profile(session, chat_id: str, role: str) -> Dict[str, Any]:
    stmt = (
        select(ActorProfile)
        .where(ActorProfile.chat_id == chat_id, ActorProfile.role == role)
        .order_by(ActorProfile.updated_at.desc())
        .limit(1)
    )
    profile = session.exec(stmt).first()
    if not profile:
        return {"role": role}
    return loads_json(profile.profile_json, {"role": role})


def _emotion_for_text(text: str) -> Dict[str, Any]:
    label, conf = rule_emotion(text)
    if conf < 0.6:
        llm = classify_emotion_llm(text)
        if llm and llm.get("label"):
            return {"label": llm["label"], "confidence": llm["confidence"], "method": "llm"}
    return {"label": label, "confidence": conf, "method": "rule"}


def _topic_for_text(text: str) -> str:
    topic = tag_topic_llm(text)
    if topic:
        return topic
    return rule_topic(text)


def _norm_text(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"([。！？!?；;])", text)
    sentences: List[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if part in "。！？!?；;":
            buf = f"{buf}{part}"
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
        else:
            buf = f"{buf}{part}" if buf else part
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


def _last_sentence(text: str) -> str:
    parts = re.split(r"[。！？!?；;]\s*", (text or "").strip())
    for part in reversed(parts):
        part = part.strip()
        if part:
            return part
    return ""


def _dedupe_phrases(text: str) -> str:
    if not text:
        return text
    phrases = (
        "你怎么想的呢",
        "你怎么想的",
        "你怎么看",
        "你觉得呢",
        "你觉得",
        "你会怎么做",
        "你会怎么办",
        "你要怎么做",
        "你要不要",
        "你愿意吗",
        "你可以吗",
    )
    for phrase in phrases:
        if text.count(phrase) > 1:
            first = text.find(phrase)
            text = f"{text[: first + len(phrase)]}{text[first + len(phrase):].replace(phrase, '')}"
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"([，,]){2,}", "，", text)
    text = re.sub(r"^[，,\s]+", "", text)
    return text.strip()


def _strip_repeated_tail(prev: str, curr: str) -> str:
    tail = _last_sentence(prev)
    if tail and len(tail) >= 4 and tail in curr:
        curr = curr.replace(tail, "", 1).strip()
    return curr


def _avoid_repeat_rule(last_npc: str) -> str:
    tail = _last_sentence(last_npc)
    if not tail:
        return ""
    return f"避免复述上一条NPC内容或末句（尤其问句）。上一条末句：{tail}"


def _sanitize_npc_reply(prev: str, curr: str, world: Dict[str, Any]) -> str:
    if not curr:
        return curr
    result = curr.strip()
    if prev:
        result = _strip_repeated_tail(prev, result)
        result = _dedupe_phrases(result)
        if _is_repetitive(prev, result):
            sentences = _split_sentences(result)
            if len(sentences) > 1:
                base = " ".join(sentences[:-1]).strip()
            else:
                base = ""
            extra = _append_progress_hint(world)
            result = f"{base} {extra}".strip() if base else extra
    return result or curr


def _is_repetitive(prev: str, curr: str) -> bool:
    a = _norm_text(prev)
    b = _norm_text(curr)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    # simple overlap ratio
    set_a = set(a)
    set_b = set(b)
    overlap = len(set_a & set_b) / max(1, len(set_a | set_b))
    return overlap >= 0.7


def _append_progress_hint(world: Dict[str, Any]) -> str:
    hooks = []
    if isinstance(world, dict):
        event_schema = world.get("event_schema") or {}
        hooks = event_schema.get("plot_hooks") or []
    if hooks:
        return f"顺带一提，我听说{hooks[0]}。要不要去看看？"
    return "要不我们换个方向：去问问镇上的老人，或者去钟楼看看？"


def _infer_desire_score(
    session,
    chat_id: str,
    text: str,
    emotion_label: str,
    base: int = 6,
) -> float:
    score = float(base)
    length = len(text or "")
    if length >= 120:
        score += 0.8
    elif length >= 60:
        score += 0.4
    elif length <= 8:
        score -= 0.4

    if "?" in text or "？" in text:
        score += 0.3

    if emotion_label in {"joy", "surprise"}:
        score += 0.6
    elif emotion_label in {"sad"}:
        score -= 0.4
    elif emotion_label in {"anger", "fear"}:
        score -= 0.2

    now = datetime.now(timezone.utc)
    last = session.exec(
        select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at.desc()).limit(1)
    ).first()
    if last and last.created_at:
        last_time = last.created_at
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        delta = (now - last_time).total_seconds()
        if delta <= 20:
            score += 0.5
        elif delta >= 120:
            score -= 0.5

    recent = session.exec(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at.desc())
        .limit(6)
    ).all()
    if recent:
        last_time = recent[-1].created_at
        if last_time:
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
            span = (now - last_time).total_seconds()
            if span <= 300:
                score += 0.4
            elif span >= 900:
                score -= 0.4

    return score


def _final_desire_score(
    session,
    chat_id: str,
    text: str,
    emotion_label: str,
    explicit: int | None,
) -> int:
    explicit_score = clamp_int(explicit if explicit is not None else 6, 1, 10)
    inferred = _infer_desire_score(session, chat_id, text, emotion_label, base=6)
    blended = explicit_score * 0.6 + inferred * 0.4
    return clamp_int(int(round(blended)), 1, 10)


@app.post("/chats/{chat_id}/message")
def chat_message(chat_id: str, payload: MessageCreate) -> Dict[str, Any]:
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        config = get_memory_config(session, chat_id, payload.role)
        last = session.exec(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.turn_index.desc())
            .limit(1)
        ).first()
        turn_index = _next_turn_index(last, payload.role)
        emotion = _emotion_for_text(payload.content)
        desire = _final_desire_score(session, chat_id, payload.content, emotion["label"], payload.desire_score)
        topic_tag = _topic_for_text(payload.content)
        message = Message(
            id=new_id(),
            chat_id=chat_id,
            role=payload.role,
            content=payload.content,
            emotion_label=emotion["label"],
            emotion_confidence=emotion["confidence"],
            emotion_method=emotion["method"],
            desire_score=desire,
            topic_tag=topic_tag,
            turn_index=turn_index,
        )
        session.add(message)
        update_memory(session, chat_id, payload.role, config["add_mode"])
        session.commit()
        _invalidate_chat_cache(chat_id)
        return {
            "id": message.id,
            "emotion_label": message.emotion_label,
            "emotion_confidence": message.emotion_confidence,
            "emotion_method": message.emotion_method,
            "topic_tag": message.topic_tag,
        }


@app.post("/chats/{chat_id}/message/stream")
def chat_message_stream(chat_id: str, payload: MessageCreate) -> StreamingResponse:
    def sse(data: Dict[str, Any]) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def event_stream():
        with get_session() as session:
            chat = session.get(Chat, chat_id)
            if not chat:
                yield sse({"type": "error", "message": "chat not found"})
                return
            world = session.get(WorldModel, chat.world_id)
            if not world:
                yield sse({"type": "error", "message": "world not found"})
                return
            config = get_memory_config(session, chat_id, payload.role)
            last = session.exec(
                select(Message)
                .where(Message.chat_id == chat_id)
                .order_by(Message.turn_index.desc())
                .limit(1)
            ).first()
            user_turn = _next_turn_index(last, payload.role)
            emotion = _emotion_for_text(payload.content)
            desire = _final_desire_score(session, chat_id, payload.content, emotion["label"], payload.desire_score)
            topic_tag = _topic_for_text(payload.content)
            user_message = Message(
                id=new_id(),
                chat_id=chat_id,
                role=payload.role,
                content=payload.content,
                emotion_label=emotion["label"],
                emotion_confidence=emotion["confidence"],
                emotion_method=emotion["method"],
                desire_score=desire,
                topic_tag=topic_tag,
                turn_index=user_turn,
            )
            session.add(user_message)
            update_memory(session, chat_id, payload.role, config["add_mode"])
            session.commit()
            _invalidate_chat_cache(chat_id)

            world_snapshot = _world_snapshot_cached(world)
            npc_profile = _actor_profile(session, chat_id, "npc")
            history = _history_cached(session, chat_id, "npc", config["search_mode"])
            last_npc = next((m["content"] for m in reversed(history) if m.get("role") == "npc"), "")
            avoid_rule = _avoid_repeat_rule(last_npc)

            if npc_profile.get("mode") == "volcengine":
                system_prompt = _volc_sp_from_profile(world_snapshot, npc_profile)
                if avoid_rule:
                    system_prompt = f"{system_prompt}\n{avoid_rule}"
                stream = generate_npc_reply_stream_with_sp(system_prompt, dumps_json(history))
            else:
                context = _build_context(world_snapshot, history, npc_profile)
                if avoid_rule:
                    context = f"{context}\n{avoid_rule}"
                stream = generate_npc_reply_stream(context)

            chunks: List[str] = []
            for delta in stream:
                if not delta:
                    continue
                chunks.append(delta)
                yield sse({"type": "delta", "content": delta})

            npc_text_raw = "".join(chunks).strip() or "(NPC) 我们继续吧。"
            npc_text = _sanitize_npc_reply(last_npc, npc_text_raw, world_snapshot)
            if npc_text != npc_text_raw and npc_text.startswith(npc_text_raw):
                extra = npc_text[len(npc_text_raw):].strip()
                if extra:
                    yield sse({"type": "delta", "content": " " + extra})
            npc_emotion = _emotion_for_text(npc_text)
            npc_topic = _topic_for_text(npc_text)
            npc_turn = _next_turn_index(user_message, "npc")
            npc_message = Message(
                id=new_id(),
                chat_id=chat_id,
                role="npc",
                content=npc_text,
                emotion_label=npc_emotion["label"],
                emotion_confidence=npc_emotion["confidence"],
                emotion_method=npc_emotion["method"],
                desire_score=None,
                topic_tag=npc_topic,
                turn_index=npc_turn,
            )
            session.add(npc_message)
            update_memory(session, chat_id, "npc", config["add_mode"])
            session.commit()
            _invalidate_chat_cache(chat_id)
            yield sse(
                {
                    "type": "done",
                    "message": {
                        "id": npc_message.id,
                        "role": npc_message.role,
                        "content": npc_message.content,
                        "emotion_label": npc_message.emotion_label,
                        "emotion_confidence": npc_message.emotion_confidence,
                        "emotion_method": npc_message.emotion_method,
                        "desire_score": npc_message.desire_score,
                        "topic_tag": npc_message.topic_tag,
                        "turn_index": npc_message.turn_index,
                        "created_at": npc_message.created_at.isoformat(),
                    },
                }
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chats/{chat_id}/auto/one")
def auto_one(chat_id: str) -> Dict[str, Any]:
    return _auto_chat(chat_id, mode="one", max_steps=1)


@app.post("/chats/{chat_id}/auto/step")
def auto_step(chat_id: str, payload: AutoChatRequest) -> Dict[str, Any]:
    return _auto_chat(chat_id, mode="step", max_steps=payload.max_steps)


@app.post("/chats/{chat_id}/auto/until")
def auto_until(chat_id: str, payload: AutoChatRequest) -> Dict[str, Any]:
    return _auto_chat(chat_id, mode="until", max_steps=payload.max_steps, desire_stop=payload.desire_stop)


@app.post("/chats/{chat_id}/auto/one/stream")
def auto_one_stream(chat_id: str) -> StreamingResponse:
    return _auto_chat_stream(chat_id, mode="one", max_steps=1, desire_stop=4)


@app.post("/chats/{chat_id}/auto/step/stream")
def auto_step_stream(chat_id: str, payload: AutoChatRequest) -> StreamingResponse:
    return _auto_chat_stream(chat_id, mode="step", max_steps=payload.max_steps, desire_stop=payload.desire_stop)


@app.post("/chats/{chat_id}/auto/until/stream")
def auto_until_stream(chat_id: str, payload: AutoChatRequest) -> StreamingResponse:
    return _auto_chat_stream(chat_id, mode="until", max_steps=payload.max_steps, desire_stop=payload.desire_stop)


@app.get("/chats/{chat_id}/stats", response_model=StatsResponse)
def chat_stats(chat_id: str) -> Dict[str, Any]:
    with get_session() as session:
        return _stats_cached(session, chat_id)


@app.get("/chats/{chat_id}/memory/config", response_model=MemoryConfigResponse)
def memory_config_get(chat_id: str) -> Dict[str, Any]:
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        config = get_memory_config(session, chat_id, "npc")
        return {
            "chat_id": chat_id,
            "add_mode": config["add_mode"],
            "search_mode": config["search_mode"],
            "supported_add_modes": supported_add_modes("npc"),
            "supported_search_modes": supported_search_modes("npc"),
        }


@app.get("/chats/{chat_id}/memory/config/user", response_model=MemoryConfigResponse)
def memory_config_get_user(chat_id: str) -> Dict[str, Any]:
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        config = get_memory_config(session, chat_id, "user")
        return {
            "chat_id": chat_id,
            "add_mode": config["add_mode"],
            "search_mode": config["search_mode"],
            "supported_add_modes": supported_add_modes("user"),
            "supported_search_modes": supported_search_modes("user"),
        }


@app.get("/chats/{chat_id}/memory/{role}")
def memory_state_get(chat_id: str, role: str) -> Dict[str, Any]:
    role = role.lower()
    if role not in ("npc", "user"):
        raise HTTPException(status_code=400, detail="role must be npc or user")
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        items = get_memory(session, chat_id, role)
        state = session.get(MemoryState, f"{chat_id}:{role}")
        mode = state.mode if state else DEFAULT_ADD_MODE
        return {"role": role, "mode": mode, "items": items}


@app.put("/chats/{chat_id}/memory/config", response_model=MemoryConfigResponse)
def memory_config_update(chat_id: str, payload: MemoryConfigUpdate) -> Dict[str, Any]:
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        add_mode = payload.add_mode
        search_mode = payload.search_mode
        if add_mode and add_mode not in supported_add_modes("npc"):
            raise HTTPException(status_code=400, detail="unsupported add_mode")
        if search_mode and search_mode not in supported_search_modes("npc"):
            raise HTTPException(status_code=400, detail="unsupported search_mode")
        config = set_memory_config(session, chat_id, add_mode, search_mode, "npc")
        update_memory(session, chat_id, "npc", config["add_mode"])
        session.commit()
        return {
            "chat_id": chat_id,
            "add_mode": config["add_mode"],
            "search_mode": config["search_mode"],
            "supported_add_modes": supported_add_modes("npc"),
            "supported_search_modes": supported_search_modes("npc"),
        }


@app.put("/chats/{chat_id}/memory/config/user", response_model=MemoryConfigResponse)
def memory_config_update_user(chat_id: str, payload: MemoryConfigUpdate) -> Dict[str, Any]:
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        add_mode = payload.add_mode
        search_mode = payload.search_mode
        if add_mode and add_mode not in supported_add_modes("user"):
            raise HTTPException(status_code=400, detail="unsupported add_mode")
        if search_mode and search_mode not in supported_search_modes("user"):
            raise HTTPException(status_code=400, detail="unsupported search_mode")
        config = set_memory_config(session, chat_id, add_mode, search_mode, "user")
        update_memory(session, chat_id, "user", config["add_mode"])
        session.commit()
        return {
            "chat_id": chat_id,
            "add_mode": config["add_mode"],
            "search_mode": config["search_mode"],
            "supported_add_modes": supported_add_modes("user"),
            "supported_search_modes": supported_search_modes("user"),
        }


@app.get("/chats/{chat_id}/session", response_model=SessionConfigResponse)
def session_config_get(chat_id: str) -> Dict[str, Any]:
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        config = session.get(SessionConfig, chat_id)
        if not config:
            config = SessionConfig(chat_id=chat_id)
            session.add(config)
            session.commit()
            session.refresh(config)
        return {"chat_id": chat_id, "session_time": config.session_time}


@app.put("/chats/{chat_id}/session", response_model=SessionConfigResponse)
def session_config_update(chat_id: str, payload: SessionConfigUpdate) -> Dict[str, Any]:
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        config = session.get(SessionConfig, chat_id)
        if not config:
            config = SessionConfig(chat_id=chat_id)
            session.add(config)
        if payload.session_time:
            config.session_time = payload.session_time
        session.commit()
        session.refresh(config)
        return {"chat_id": chat_id, "session_time": config.session_time}


@app.get("/chats/{chat_id}/actors")
def chat_actors(chat_id: str) -> Dict[str, Any]:
    def latest_profile(role: str) -> Dict[str, Any] | None:
        stmt = (
            select(ActorProfile)
            .where(ActorProfile.chat_id == chat_id, ActorProfile.role == role)
            .order_by(ActorProfile.updated_at.desc())
            .limit(1)
        )
        profile = session.exec(stmt).first()
        if not profile:
            return None
        return {"id": profile.id, "profile": loads_json(profile.profile_json, {"role": role})}

    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        return {"npc": latest_profile("npc"), "user": latest_profile("user")}


@app.get("/chats/{chat_id}/export")
def chat_export(chat_id: str) -> Dict[str, Any]:
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        world = session.get(WorldModel, chat.world_id)
        world_snapshot = _world_snapshot_cached(world) if world else {}
        world_events = session.exec(
            select(WorldEvent).where(WorldEvent.world_id == chat.world_id).order_by(WorldEvent.version)
        ).all()
        actors = session.exec(
            select(ActorProfile).where(ActorProfile.chat_id == chat_id).order_by(ActorProfile.updated_at)
        ).all()
        messages = session.exec(
            select(Message).where(Message.chat_id == chat_id).order_by(Message.created_at)
        ).all()
        memory_states = session.exec(
            select(MemoryState).where(MemoryState.chat_id == chat_id).order_by(MemoryState.updated_at)
        ).all()
        memory_config = session.get(MemoryConfig, chat_id)
        session_config = session.get(SessionConfig, chat_id)

        return {
            "chat": {
                "id": chat.id,
                "world_id": chat.world_id,
                "user_id": chat.user_id,
                "created_at": chat.created_at.isoformat(),
            },
            "world": {
                "id": world.id if world else None,
                "name": world.name if world else None,
                "version": world.version if world else None,
                "snapshot": _normalize_world(world_snapshot),
            },
            "world_events": [
                {
                    "id": e.id,
                    "version": e.version,
                    "event_type": e.event_type,
                    "payload": loads_json(e.payload_json, {}),
                    "created_at": e.created_at.isoformat(),
                }
                for e in world_events
            ],
            "actors": [
                {
                    "id": a.id,
                    "role": a.role,
                    "profile": loads_json(a.profile_json, {}),
                    "updated_at": a.updated_at.isoformat(),
                }
                for a in actors
            ],
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "emotion_label": m.emotion_label,
                    "emotion_confidence": m.emotion_confidence,
                    "emotion_method": m.emotion_method,
                    "desire_score": m.desire_score,
                    "topic_tag": m.topic_tag,
                    "turn_index": m.turn_index,
                    "created_at": m.created_at.isoformat(),
                }
                for m in messages
            ],
            "memory_states": [
                {
                    "id": s.id,
                    "role": s.role,
                    "mode": s.mode,
                    "memory": loads_json(s.memory_json, {}),
                    "updated_at": s.updated_at.isoformat(),
                }
                for s in memory_states
            ],
            "memory_config": {
                "add_mode": memory_config.add_mode if memory_config else "last_60",
                "search_mode": memory_config.search_mode if memory_config else "last_60",
            },
            "session_config": {
                "session_time": session_config.session_time if session_config else None,
                "updated_at": session_config.updated_at.isoformat() if session_config else None,
            },
            "stats": _stats_cached(session, chat_id),
        }


@app.patch("/actor/{actor_id}")
def actor_update(actor_id: str, payload: ActorUpdate) -> Dict[str, Any]:
    with get_session() as session:
        profile = session.get(ActorProfile, actor_id)
        if not profile:
            raise HTTPException(status_code=404, detail="actor not found")
        profile.profile_json = dumps_json(payload.profile)
        session.add(profile)
        session.commit()
        session.refresh(profile)
        return {"id": profile.id}


@app.get("/actor/{actor_id}")
def actor_get(actor_id: str) -> Dict[str, Any]:
    with get_session() as session:
        profile = session.get(ActorProfile, actor_id)
        if not profile:
            raise HTTPException(status_code=404, detail="actor not found")
        return {
            "id": profile.id,
            "role": profile.role,
            "profile": loads_json(profile.profile_json, {}),
            "updated_at": profile.updated_at.isoformat(),
        }


@app.post("/actor")
def actor_create(payload: ActorUpdate) -> Dict[str, Any]:
    actor = ActorProfile(
        id=new_id(),
        chat_id=payload.profile.get("chat_id", ""),
        role=payload.profile.get("role", "npc"),
        profile_json=dumps_json(payload.profile),
    )
    with get_session() as session:
        session.add(actor)
        session.commit()
        session.refresh(actor)
    return {"id": actor.id}


def _auto_chat(chat_id: str, mode: str, max_steps: int, desire_stop: int = 5) -> Dict[str, Any]:
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        world = session.get(WorldModel, chat.world_id)
        if not world:
            raise HTTPException(status_code=404, detail="world not found")
        config = get_memory_config(session, chat_id, "npc")
        user_config = get_memory_config(session, chat_id, "user")
        world_snapshot = _world_snapshot_cached(world)
        npc_profile = _actor_profile(session, chat_id, "npc")
        user_profile = _actor_profile(session, chat_id, "user")
        history = _history_cached(session, chat_id, "npc", config["search_mode"])
        last = session.exec(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.turn_index.desc())
            .limit(1)
        ).first()
        last_msg = last
        created = []
        steps = 0
        while steps < max_steps:
            pre_user_desire: float | None = None
            if last_msg and last_msg.role == "npc":
                user_context = _build_user_context(_recent_context(session, chat_id), user_profile)
                user_data = generate_user_reply(user_context)
                user_text = user_data.get("content", "")
                try:
                    desire_value = int(user_data.get("desire_score", 6))
                except Exception:
                    desire_value = 6
                emotion = _emotion_for_text(user_text)
                pre_user_desire = _final_desire_score(session, chat_id, user_text, emotion["label"], desire_value)
                topic_tag = _topic_for_text(user_text)
                user_turn = _next_turn_index(last_msg, "user")
                user_msg = Message(
                    id=new_id(),
                    chat_id=chat_id,
                    role="user",
                    content=user_text,
                    emotion_label=emotion["label"],
                    emotion_confidence=emotion["confidence"],
                    emotion_method=emotion["method"],
                    desire_score=pre_user_desire,
                    topic_tag=topic_tag,
                    turn_index=user_turn,
                )
                session.add(user_msg)
                update_memory(session, chat_id, "user", user_config["add_mode"])
                session.commit()
                _invalidate_chat_cache(chat_id)
                created.append(user_msg)
                last_msg = user_msg
                history.append({"role": "user", "content": user_text})

            last_npc = next((m["content"] for m in reversed(history) if m.get("role") == "npc"), "")
            avoid_rule = _avoid_repeat_rule(last_npc)
            if npc_profile.get("mode") == "volcengine":
                system_prompt = _volc_sp_from_profile(world_snapshot, npc_profile)
                if avoid_rule:
                    system_prompt = f"{system_prompt}\n{avoid_rule}"
                npc_text = generate_npc_reply_with_sp(system_prompt, dumps_json(history))
            else:
                context = _build_context(world_snapshot, history, npc_profile)
                if avoid_rule:
                    context = f"{context}\n{avoid_rule}"
                npc_text = generate_npc_reply(context)
            npc_text = _sanitize_npc_reply(last_npc, npc_text, world_snapshot)
            emotion = _emotion_for_text(npc_text)
            topic_tag = _topic_for_text(npc_text)
            npc_turn = _next_turn_index(last_msg, "npc")
            npc_msg = Message(
                id=new_id(),
                chat_id=chat_id,
                role="npc",
                content=npc_text,
                emotion_label=emotion["label"],
                emotion_confidence=emotion["confidence"],
                emotion_method=emotion["method"],
                desire_score=None,
                topic_tag=topic_tag,
                turn_index=npc_turn,
            )
            session.add(npc_msg)
            update_memory(session, chat_id, "npc", config["add_mode"])
            session.commit()
            _invalidate_chat_cache(chat_id)
            created.append(npc_msg)
            if mode == "one":
                break
            last_msg = npc_msg
            history.append({"role": "npc", "content": npc_text})
            if pre_user_desire is not None:
                if mode == "until" and pre_user_desire < desire_stop:
                    break
                steps += 1
                continue
            user_context = _build_user_context(_recent_context(session, chat_id), user_profile)
            user_data = generate_user_reply(user_context)
            user_text = user_data.get("content", "")
            try:
                desire_value = int(user_data.get("desire_score", 6))
            except Exception:
                desire_value = 6
            desire = _final_desire_score(session, chat_id, user_text, emotion["label"], desire_value)
            emotion = _emotion_for_text(user_text)
            topic_tag = _topic_for_text(user_text)
            user_turn = _next_turn_index(last_msg, "user")
            user_msg = Message(
                id=new_id(),
                chat_id=chat_id,
                role="user",
                content=user_text,
                emotion_label=emotion["label"],
                emotion_confidence=emotion["confidence"],
                emotion_method=emotion["method"],
                desire_score=desire,
                topic_tag=topic_tag,
                turn_index=user_turn,
            )
            session.add(user_msg)
            update_memory(session, chat_id, "user", user_config["add_mode"])
            session.commit()
            _invalidate_chat_cache(chat_id)
            created.append(user_msg)
            history.append({"role": "user", "content": user_text})
            if mode == "until" and desire < desire_stop:
                break
            last_msg = user_msg
            steps += 1
        return {
            "created": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "emotion_label": m.emotion_label,
                    "emotion_confidence": m.emotion_confidence,
                    "emotion_method": m.emotion_method,
                    "desire_score": m.desire_score,
                    "topic_tag": m.topic_tag,
                    "turn_index": m.turn_index,
                }
                for m in created
            ]
        }


def _auto_chat_stream(chat_id: str, mode: str, max_steps: int, desire_stop: int = 4) -> StreamingResponse:
    def sse(data: Dict[str, Any]) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def event_stream():
        with get_session() as session:
            chat = session.get(Chat, chat_id)
            if not chat:
                yield sse({"type": "error", "message": "chat not found"})
                return
            world = session.get(WorldModel, chat.world_id)
            if not world:
                yield sse({"type": "error", "message": "world not found"})
                return
            config = get_memory_config(session, chat_id, "npc")
            user_config = get_memory_config(session, chat_id, "user")
            world_snapshot = _world_snapshot_cached(world)
            npc_profile = _actor_profile(session, chat_id, "npc")
            user_profile = _actor_profile(session, chat_id, "user")
            history = _history_cached(session, chat_id, "npc", config["search_mode"])
            last = session.exec(
                select(Message)
                .where(Message.chat_id == chat_id)
                .order_by(Message.turn_index.desc())
                .limit(1)
            ).first()
            last_msg = last
            steps = 0

            while steps < max_steps:
                pre_user_desire: float | None = None
                if last_msg and last_msg.role == "npc":
                    user_context = _build_user_context(_recent_context(session, chat_id), user_profile)
                    user_data = generate_user_reply(user_context)
                    user_text = user_data.get("content", "")
                    try:
                        desire_value = int(user_data.get("desire_score", 6))
                    except Exception:
                        desire_value = 6
                    emotion = _emotion_for_text(user_text)
                    pre_user_desire = _final_desire_score(session, chat_id, user_text, emotion["label"], desire_value)
                    topic_tag = _topic_for_text(user_text)
                    user_turn = _next_turn_index(last_msg, "user")
                    user_msg = Message(
                        id=new_id(),
                        chat_id=chat_id,
                        role="user",
                        content=user_text,
                        emotion_label=emotion["label"],
                        emotion_confidence=emotion["confidence"],
                        emotion_method=emotion["method"],
                        desire_score=pre_user_desire,
                        topic_tag=topic_tag,
                        turn_index=user_turn,
                    )
                    session.add(user_msg)
                    update_memory(session, chat_id, "user", user_config["add_mode"])
                    session.commit()
                    _invalidate_chat_cache(chat_id)
                    yield sse(
                        {
                            "type": "user_done",
                            "message": {
                                "id": user_msg.id,
                                "role": user_msg.role,
                                "content": user_msg.content,
                                "emotion_label": user_msg.emotion_label,
                                "emotion_confidence": user_msg.emotion_confidence,
                                "emotion_method": user_msg.emotion_method,
                                "desire_score": user_msg.desire_score,
                                "topic_tag": user_msg.topic_tag,
                                "turn_index": user_msg.turn_index,
                                "created_at": user_msg.created_at.isoformat(),
                            },
                        }
                    )
                    last_msg = user_msg
                    history.append({"role": "user", "content": user_text})

                last_npc = next((m["content"] for m in reversed(history) if m.get("role") == "npc"), "")
                avoid_rule = _avoid_repeat_rule(last_npc)
                if npc_profile.get("mode") == "volcengine":
                    system_prompt = _volc_sp_from_profile(world_snapshot, npc_profile)
                    if avoid_rule:
                        system_prompt = f"{system_prompt}\n{avoid_rule}"
                    stream = generate_npc_reply_stream_with_sp(system_prompt, dumps_json(history))
                else:
                    context = _build_context(world_snapshot, history, npc_profile)
                    if avoid_rule:
                        context = f"{context}\n{avoid_rule}"
                    stream = generate_npc_reply_stream(context)

                chunks: List[str] = []
                for delta in stream:
                    if not delta:
                        continue
                    chunks.append(delta)
                    yield sse({"type": "npc_delta", "content": delta})

                npc_text_raw = "".join(chunks).strip() or "(NPC) 我们继续吧。"
                npc_text = _sanitize_npc_reply(last_npc, npc_text_raw, world_snapshot)
                if npc_text != npc_text_raw and npc_text.startswith(npc_text_raw):
                    extra = npc_text[len(npc_text_raw):].strip()
                    if extra:
                        yield sse({"type": "npc_delta", "content": " " + extra})

                emotion = _emotion_for_text(npc_text)
                topic_tag = _topic_for_text(npc_text)
                npc_turn = _next_turn_index(last_msg, "npc")
                npc_msg = Message(
                    id=new_id(),
                    chat_id=chat_id,
                    role="npc",
                    content=npc_text,
                    emotion_label=emotion["label"],
                    emotion_confidence=emotion["confidence"],
                    emotion_method=emotion["method"],
                    desire_score=None,
                    topic_tag=topic_tag,
                    turn_index=npc_turn,
                )
                session.add(npc_msg)
                update_memory(session, chat_id, "npc", config["add_mode"])
                session.commit()
                _invalidate_chat_cache(chat_id)
                yield sse(
                    {
                        "type": "npc_done",
                        "message": {
                            "id": npc_msg.id,
                            "role": npc_msg.role,
                            "content": npc_msg.content,
                            "emotion_label": npc_msg.emotion_label,
                            "emotion_confidence": npc_msg.emotion_confidence,
                            "emotion_method": npc_msg.emotion_method,
                            "desire_score": npc_msg.desire_score,
                            "topic_tag": npc_msg.topic_tag,
                            "turn_index": npc_msg.turn_index,
                            "created_at": npc_msg.created_at.isoformat(),
                        },
                    }
                )

                if mode == "one":
                    break

                last_msg = npc_msg
                history.append({"role": "npc", "content": npc_text})
                if pre_user_desire is not None:
                    if mode == "until" and pre_user_desire < desire_stop:
                        break
                    steps += 1
                    continue
                user_context = _build_user_context(_recent_context(session, chat_id), user_profile)
                user_data = generate_user_reply(user_context)
                user_text = user_data.get("content", "")
                try:
                    desire_value = int(user_data.get("desire_score", 6))
                except Exception:
                    desire_value = 6
                emotion = _emotion_for_text(user_text)
                desire = _final_desire_score(session, chat_id, user_text, emotion["label"], desire_value)
                topic_tag = _topic_for_text(user_text)
                user_turn = _next_turn_index(last_msg, "user")
                user_msg = Message(
                    id=new_id(),
                    chat_id=chat_id,
                    role="user",
                    content=user_text,
                    emotion_label=emotion["label"],
                    emotion_confidence=emotion["confidence"],
                    emotion_method=emotion["method"],
                    desire_score=desire,
                    topic_tag=topic_tag,
                    turn_index=user_turn,
                )
                session.add(user_msg)
                update_memory(session, chat_id, "user", user_config["add_mode"])
                session.commit()
                _invalidate_chat_cache(chat_id)
                yield sse(
                    {
                        "type": "user_done",
                        "message": {
                            "id": user_msg.id,
                            "role": user_msg.role,
                            "content": user_msg.content,
                            "emotion_label": user_msg.emotion_label,
                            "emotion_confidence": user_msg.emotion_confidence,
                            "emotion_method": user_msg.emotion_method,
                            "desire_score": user_msg.desire_score,
                            "topic_tag": user_msg.topic_tag,
                            "turn_index": user_msg.turn_index,
                            "created_at": user_msg.created_at.isoformat(),
                        },
                    }
                )
                history.append({"role": "user", "content": user_text})
                if mode == "until" and desire < desire_stop:
                    break
                last_msg = user_msg
                steps += 1

            yield sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
