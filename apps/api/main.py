from __future__ import annotations

from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
)
from .utils import new_id, new_session_id, dumps_json, loads_json, clamp_int
from .llm import (
    generate_world,
    generate_npc_reply,
    generate_npc_reply_with_sp,
    generate_user_reply,
    classify_emotion_llm,
    tag_topic_llm,
)
from .memory import (
    rule_emotion,
    rule_topic,
    update_memory,
    search_memory,
    get_memory_config,
    set_memory_config,
    supported_add_modes,
    supported_search_modes,
)
from .stats import compute_stats

app = FastAPI(title="AI Town Lite")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        snapshot = _normalize_world(loads_json(world.snapshot_json, {}))
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


def _volc_sp_from_profile(world: Dict[str, Any], npc_profile: Dict[str, Any]) -> str:
    role_name = npc_profile.get("role_name", "NPC")
    user_name = npc_profile.get("user_name", "用户")
    init_role_sp = npc_profile.get("init_role_sp", "你是一个有自己背景与动机的角色。")
    user_info = npc_profile.get("user_info", "你和用户正在进行日常对话。")
    golden_sp = npc_profile.get("golden_sp", "")
    defaults = [
        "你使用口语进行表达，必要时可用括号描述动作和情绪。",
        "你需要尽可能引导用户跟你进行交流，你不应该表现地太AI。",
        "每次回复推进剧情，提出问题或下一步行动。",
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


@app.post("/chats/{chat_id}/message")
def chat_message(chat_id: str, payload: MessageCreate) -> Dict[str, Any]:
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        config = get_memory_config(session, chat_id)
        last = session.exec(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.turn_index.desc())
            .limit(1)
        ).first()
        turn_index = _next_turn_index(last, payload.role)
        emotion = _emotion_for_text(payload.content)
        desire = payload.desire_score if payload.desire_score is not None else 6
        desire = clamp_int(desire, 1, 10)
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
        return {
            "id": message.id,
            "emotion_label": message.emotion_label,
            "emotion_confidence": message.emotion_confidence,
            "emotion_method": message.emotion_method,
            "topic_tag": message.topic_tag,
        }


@app.post("/chats/{chat_id}/auto/one")
def auto_one(chat_id: str) -> Dict[str, Any]:
    return _auto_chat(chat_id, mode="one", max_steps=1)


@app.post("/chats/{chat_id}/auto/step")
def auto_step(chat_id: str, payload: AutoChatRequest) -> Dict[str, Any]:
    return _auto_chat(chat_id, mode="step", max_steps=payload.max_steps)


@app.post("/chats/{chat_id}/auto/until")
def auto_until(chat_id: str, payload: AutoChatRequest) -> Dict[str, Any]:
    return _auto_chat(chat_id, mode="until", max_steps=payload.max_steps, desire_stop=payload.desire_stop)


@app.get("/chats/{chat_id}/stats", response_model=StatsResponse)
def chat_stats(chat_id: str) -> Dict[str, Any]:
    with get_session() as session:
        return compute_stats(session, chat_id)


@app.get("/chats/{chat_id}/memory/config", response_model=MemoryConfigResponse)
def memory_config_get(chat_id: str) -> Dict[str, Any]:
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        config = get_memory_config(session, chat_id)
        return {
            "chat_id": chat_id,
            "add_mode": config["add_mode"],
            "search_mode": config["search_mode"],
            "supported_add_modes": supported_add_modes(),
            "supported_search_modes": supported_search_modes(),
        }


@app.put("/chats/{chat_id}/memory/config", response_model=MemoryConfigResponse)
def memory_config_update(chat_id: str, payload: MemoryConfigUpdate) -> Dict[str, Any]:
    with get_session() as session:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="chat not found")
        add_mode = payload.add_mode
        search_mode = payload.search_mode
        if add_mode and add_mode not in supported_add_modes():
            raise HTTPException(status_code=400, detail="unsupported add_mode")
        if search_mode and search_mode not in supported_search_modes():
            raise HTTPException(status_code=400, detail="unsupported search_mode")
        config = set_memory_config(session, chat_id, add_mode, search_mode)
        update_memory(session, chat_id, "npc", config["add_mode"])
        update_memory(session, chat_id, "user", config["add_mode"])
        session.commit()
        return {
            "chat_id": chat_id,
            "add_mode": config["add_mode"],
            "search_mode": config["search_mode"],
            "supported_add_modes": supported_add_modes(),
            "supported_search_modes": supported_search_modes(),
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
        world_snapshot = loads_json(world.snapshot_json, {}) if world else {}
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
            "stats": compute_stats(session, chat_id),
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
        config = get_memory_config(session, chat_id)
        world_snapshot = loads_json(world.snapshot_json, {})
        npc_profile = _actor_profile(session, chat_id, "npc")
        user_profile = _actor_profile(session, chat_id, "user")
        history = _recent_history(session, chat_id, "npc", config["search_mode"])
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
            if npc_profile.get("mode") == "volcengine":
                system_prompt = _volc_sp_from_profile(world_snapshot, npc_profile)
                npc_text = generate_npc_reply_with_sp(system_prompt, dumps_json(history))
            else:
                context = _build_context(world_snapshot, history, npc_profile)
                npc_text = generate_npc_reply(context)
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
            created.append(npc_msg)
            if mode == "one":
                break
            last_msg = npc_msg
            history.append({"role": "npc", "content": npc_text})
            user_context = _build_context(world_snapshot, history, user_profile)
            user_data = generate_user_reply(user_context)
            user_text = user_data.get("content", "")
            try:
                desire_value = int(user_data.get("desire_score", 6))
            except Exception:
                desire_value = 6
            desire = clamp_int(desire_value, 1, 10)
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
            update_memory(session, chat_id, "user", config["add_mode"])
            session.commit()
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
