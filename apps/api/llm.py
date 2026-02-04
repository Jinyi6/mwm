from __future__ import annotations

import os
import json
import logging
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

logger = logging.getLogger("mwm.llm")


def _snip(text: str, limit: int = 50) -> str:
    text = (text or "").replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."

def _chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = 0
    zh = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if "\u4e00" <= ch <= "\u9fff":
            zh += 1
    return zh / max(1, total)


def _mostly_chinese(text: str, threshold: float = 0.2) -> bool:
    return _chinese_ratio(text) >= threshold


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
    if messages:
        logger.info("LLM JSON request model=%s input=%s", model, _snip(messages[-1].get("content", "")))
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    content = resp.choices[0].message.content or "{}"
    logger.info("LLM JSON response model=%s output=%s", model, _snip(content))
    try:
        return json.loads(content)
    except Exception:
        return None


def generate_world(seed_prompt: str, style: str) -> Dict[str, Any]:
    def _request(extra_rule: str = "") -> Optional[Dict[str, Any]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a world model generator for a narrative game. "
                    "Output JSON only with rich, concrete details. "
                    "All output must be in Chinese."
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
                    "- scene.setting 2-4 sentences with concrete sensory details.\n"
                    "- scene.visuals include 3-6 vivid visual cues.\n"
                    "- Keep strings in Chinese, cinematic and playable.\n"
                    "{extra}\n"
                    "- Output JSON only."
                ).format(style=style, seed=seed_prompt or "none", extra=extra_rule),
            },
        ]
        return _chat_json(MODEL_NPC, messages, OPENAI_API_KEY_NPC, OPENAI_BASE_URL_NPC)

    data = _request()
    if data:
        scene_setting = str((data.get("scene") or {}).get("setting") or "")
        if _mostly_chinese(scene_setting):
            return data
        retry = _request("IMPORTANT: 必须使用中文作答，不要输出任何英文。")
        if retry:
            retry_setting = str((retry.get("scene") or {}).get("setting") or "")
            if _mostly_chinese(retry_setting):
                return retry
            scene = retry.get("scene") or {}
            scene["setting"] = ""
            if isinstance(scene.get("visuals"), str) or isinstance(scene.get("visuals"), list):
                scene["visuals"] = []
            retry["scene"] = scene
            return retry
    return {
        "scene": {
            "ontology": {"user_profile": {}, "user_state_machine": {}},
            "setting": "一座现代小镇，空气里有潮湿的泥土味，街道灯光柔和，人群稀疏。",
        },
        "event_schema": {"plot_hooks": []},
        "norms": ["Stay in character", "Keep the tone modern and cinematic"],
        "beliefs": [],
        "memory": {"semantic": [], "episodic": [], "working": []},
    }


def generate_npc_profile(world: Dict[str, Any], style: str = "modern") -> Dict[str, Any]:
    def _request(extra_rule: str = "") -> Optional[Dict[str, Any]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You create a concise NPC profile for a narrative chat game. "
                    "Output JSON only. All output must be in Chinese."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Based on the world snapshot below, generate an NPC profile JSON with fields:\n"
                    "- role_name (string, 2-4 chars)\n"
                    "- user_name (string, 2-4 chars)\n"
                    "- init_role_sp (string, 1-2 sentences)\n"
                    "- user_info (string, 1-2 sentences)\n"
                    "- golden_sp (string, 3-6 short bullet-like lines, no numbering)\n"
                    "- notes (array of 3-6 short strings)\n"
                    "- mode (string, default \"default\")\n\n"
                    "Rules:\n"
                    "- Keep Chinese only, vivid but practical.\n"
                    "- The NPC should fit the world tone and scene.\n"
                    "- Avoid generic lines like “你怎么想/你怎么看/你觉得呢”.\n"
                    f"{extra_rule}\n"
                    "- Output JSON only.\n\n"
                    f"Style: {style}\n"
                    f"World: {json.dumps(world, ensure_ascii=False)}"
                ),
            },
        ]
        return _chat_json(MODEL_NPC, messages, OPENAI_API_KEY_NPC, OPENAI_BASE_URL_NPC)

    data = _request()
    if data:
        init_sp = str(data.get("init_role_sp") or "")
        if _mostly_chinese(init_sp):
            return data
        retry = _request("IMPORTANT: 必须使用中文作答，不要输出任何英文。")
        if retry:
            return retry
    return {
        "mode": "default",
        "role_name": "林一",
        "user_name": "小周",
        "init_role_sp": "简介：你是林一，一位自由插画师。性格开朗，待人温柔体贴。",
        "user_info": "用户叫小周，是你的未婚妻，你通常称呼她为宝贝或周周。",
        "golden_sp": "你使用口语表达，必要时可用括号描述动作和情绪。\n每次回复推进剧情并给出具体行动建议。\n避免重复上一轮问题或内容。",
        "notes": [],
    }


def generate_npc_reply(context: str) -> str:
    return generate_npc_reply_with_sp(
        (
            "你是故事中的NPC，请以口语表达，必要时可用括号描述动作或情绪。"
            "你需要尽可能引导用户交流、推动剧情发展，避免闲聊空转或泛泛而谈。"
            "不要表现得像AI。每次回复至少包含一个线索或问题，引导用户参与。"
            "优先抛出具体线索、选择或行动建议，让用户更快进入剧情。"
            "避免重复上一轮NPC问题或同义复述；若用户已回答，请确认并给出新进展。"
            "避免使用“你决定怎么做”“你会怎么做”这类泛问句，改用具体选项或行动建议。"
            "仅使用中文回答。"
            "回复1-3句，保持节奏感与画面感。"
        ),
        context,
    )


def generate_npc_reply_with_sp(system_prompt: str, user_content: str) -> str:
    default_rules = (
        "你使用口语进行表达，必要时可用括号描述动作和情绪。"
        "你需要尽可能引导用户跟你进行交流，你不应该表现地太AI。"
        "每次回复推进剧情，提出问题或下一步行动。"
        "避免重复上一轮NPC内容或问句；若用户已回应，请确认并给出新进展。"
        "避免使用“你怎么想/你怎么看/你觉得呢”等泛问句，改用具体行动或选择。"
        "仅使用中文回答。"
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
    logger.info("NPC request model=%s input=%s", MODEL_NPC, _snip(user_content))
    resp = client.chat.completions.create(
        model=MODEL_NPC,
        messages=messages,
        temperature=0.8,
    )
    content = (resp.choices[0].message.content or "").strip() or "(NPC) 我们继续吧。"
    logger.info("NPC response model=%s output=%s", MODEL_NPC, _snip(content))
    return content


def generate_npc_reply_stream_with_sp(system_prompt: str, user_content: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    client = _client(OPENAI_API_KEY_NPC, OPENAI_BASE_URL_NPC)
    if client is None:
        yield "(NPC) 我们继续吧。"
        return
    logger.info("NPC stream request model=%s input=%s", MODEL_NPC, _snip(user_content))
    stream = client.chat.completions.create(
        model=MODEL_NPC,
        messages=messages,
        temperature=0.8,
        stream=True,
    )
    buffer = ""
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            buffer += delta
            yield delta
    if buffer:
        logger.info("NPC stream response model=%s output=%s", MODEL_NPC, _snip(buffer))


def generate_npc_reply_stream(context: str):
    return generate_npc_reply_stream_with_sp(
        (
            "你是故事中的NPC，请以口语表达，必要时可用括号描述动作或情绪。"
            "你需要尽可能引导用户交流、推动剧情发展，避免闲聊空转或泛泛而谈。"
            "不要表现得像AI。每次回复包含推进剧情的线索或问题。"
            "避免重复上一轮NPC内容或问句；若用户已回应，请确认并给出新进展。"
            "避免使用“你怎么想/你怎么看/你觉得呢”等泛问句，改用具体行动或选择。"
            "仅使用中文回答。"
            "回复1-3句，保持节奏感与画面感。"
        ),
        context,
    )


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
