from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

import httpx
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
MODEL_NPC = os.getenv("OPENAI_MODEL_NPC", "gpt-4o-mini")
MODEL_USER = os.getenv("OPENAI_MODEL_USER", "gpt-4o-mini")
MODEL_CLASSIFIER = os.getenv("OPENAI_MODEL_CLASSIFIER", "gpt-4o-mini")

OPENAI_API_KEY_NPC = os.getenv("OPENAI_API_KEY_NPC", "") or OPENAI_API_KEY
OPENAI_API_KEY_USER = os.getenv("OPENAI_API_KEY_USER", "") or OPENAI_API_KEY
OPENAI_API_KEY_CLASSIFIER = os.getenv("OPENAI_API_KEY_CLASSIFIER", "") or OPENAI_API_KEY

OPENAI_BASE_URL_NPC = os.getenv("OPENAI_BASE_URL_NPC", "") or OPENAI_BASE_URL
OPENAI_BASE_URL_USER = os.getenv("OPENAI_BASE_URL_USER", "") or OPENAI_BASE_URL
OPENAI_BASE_URL_CLASSIFIER = os.getenv("OPENAI_BASE_URL_CLASSIFIER", "") or OPENAI_BASE_URL


def _client(api_key: str, base_url: str) -> Optional[OpenAI]:
    if not api_key:
        return None
    http_client = httpx.Client()
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    return OpenAI(api_key=api_key, http_client=http_client)


def _chat_json(model: str, messages: List[Dict[str, str]], api_key: str, base_url: str) -> Optional[Dict[str, Any]]:
    client = _client(api_key, base_url)
    if client is None:
        return None
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except Exception:
        return None


def generate_world(seed_prompt: str, style: str) -> Dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a world model generator for a narrative game. "
                "Output JSON only with rich, concrete details."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create a detailed world model JSON with fields: "
                "scene, event_schema, norms, beliefs, memory. "
                "Style: {style}. Seed: {seed}.\n\n"
                "Schema guide:\n"
                "- scene: {{setting, era, time_of_day, weather, mood, location, visuals, tension, theme, ontology}}\n"
                "- ontology: {{user_profile, user_state_machine}}\n"
                "- event_schema: {{plot_hooks, active_events, triggers}}\n"
                "- norms: array of explicit rules\n"
                "- beliefs: array of subjective beliefs\n"
                "- memory: {{semantic, episodic, working}}\n\n"
                "Requirements:\n"
                "- plot_hooks: 6-10 items, each is a short but vivid hook\n"
                "- active_events: 2-4 items, currently unfolding events\n"
                "- semantic: 6-10 stable facts\n"
                "- episodic: 3-6 recent story beats\n"
                "- working: 3-5 immediate goals or tensions\n"
                "- Keep strings in Chinese, cinematic and playable.\n"
                "- Output JSON only."
            ).format(style=style, seed=seed_prompt or "none"),
        },
    ]
    data = _chat_json(MODEL_NPC, messages, OPENAI_API_KEY_NPC, OPENAI_BASE_URL_NPC)
    if data:
        return data
    return {
        "scene": {
            "ontology": {"user_profile": {}, "user_state_machine": {}},
            "setting": "A modern, cinematic town.",
        },
        "event_schema": {"plot_hooks": []},
        "norms": ["Stay in character", "Keep the tone modern and cinematic"],
        "beliefs": [],
        "memory": {"semantic": [], "episodic": [], "working": []},
    }


def generate_npc_reply(context: str) -> str:
    return generate_npc_reply_with_sp(
        (
            "你是故事中的NPC，请以口语表达，必要时可用括号描述动作或情绪。"
            "你需要尽可能引导用户交流、推动剧情发展，避免闲聊空转或泛泛而谈。"
            "不要表现得像AI。每次回复包含推进剧情的线索或问题。"
            "回复1-3句，保持节奏感与画面感。"
        ),
        context,
    )


def generate_npc_reply_with_sp(system_prompt: str, user_content: str) -> str:
    default_rules = (
        "你使用口语进行表达，必要时可用括号描述动作和情绪。"
        "你需要尽可能引导用户跟你进行交流，你不应该表现地太AI。"
        "每次回复推进剧情，提出问题或下一步行动。"
    )
    if default_rules not in system_prompt:
        system_prompt = f"{system_prompt}\n{default_rules}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    client = _client(OPENAI_API_KEY_NPC, OPENAI_BASE_URL_NPC)
    if client is None:
        return "(NPC) 我们继续吧。"
    resp = client.chat.completions.create(
        model=MODEL_NPC,
        messages=messages,
        temperature=0.8,
    )
    return (resp.choices[0].message.content or "").strip() or "(NPC) 我们继续吧。"


def generate_user_reply(context: str) -> Dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "你模拟真实用户，尽量拟人化、自然地回应对话。"
                "不要刻意推动剧情，重在真实反应与情感表达。"
                "输出 JSON: {content, desire_score}。content 1-3句。"
            ),
        },
        {"role": "user", "content": context},
    ]
    data = _chat_json(MODEL_USER, messages, OPENAI_API_KEY_USER, OPENAI_BASE_URL_USER)
    if data and isinstance(data.get("content"), str):
        return {
            "content": data.get("content", ""),
            "desire_score": int(data.get("desire_score", 6)),
        }
    return {"content": "(用户) 好的。", "desire_score": 6}


def classify_emotion_llm(text: str) -> Optional[Dict[str, Any]]:
    messages = [
        {"role": "system", "content": "Classify emotion. Output JSON: {label, confidence}."},
        {"role": "user", "content": text},
    ]
    data = _chat_json(MODEL_CLASSIFIER, messages, OPENAI_API_KEY_CLASSIFIER, OPENAI_BASE_URL_CLASSIFIER)
    if not data:
        return None
    return {
        "label": data.get("label"),
        "confidence": float(data.get("confidence", 0.5)),
    }


def tag_topic_llm(text: str) -> Optional[str]:
    messages = [
        {"role": "system", "content": "Assign a short topic tag (1-3 words). Output JSON: {topic}."},
        {"role": "user", "content": text},
    ]
    data = _chat_json(MODEL_CLASSIFIER, messages, OPENAI_API_KEY_CLASSIFIER, OPENAI_BASE_URL_CLASSIFIER)
    if not data:
        return None
    topic = data.get("topic")
    if isinstance(topic, str) and topic.strip():
        return topic.strip()
    return None
