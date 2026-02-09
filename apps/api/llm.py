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

DIALOGUE_STYLE_RULES: Dict[str, str] = {
    "natural": "风格=自然聊天：像熟悉的人日常交流，轻松自然，少用指令和追问。",
    "balanced": "风格=平衡推进：自然聊天为主，适度推进剧情，每2-3轮推进一个剧情点。",
    "assertive": "风格=强推进：节奏更紧，主动推进剧情并明确下一步行动。",
    "cinematic": "风格=电影叙事：加强场景感和镜头感，用细节推动剧情。",
    "warm": "风格=温柔陪伴：先共情与安抚，再自然带出线索和行动建议。",
    "suspense": "风格=悬疑推进：保留悬念与反转信息，逐步揭示关键线索。",
    "playful": "风格=轻松逗趣：语气有趣不油腻，在玩笑中自然推进剧情。",
}
DEFAULT_DIALOGUE_STYLE = "balanced"


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


def normalize_dialogue_style(value: Optional[str]) -> str:
    style = (value or "").strip().lower()
    if style in DIALOGUE_STYLE_RULES:
        return style
    return DEFAULT_DIALOGUE_STYLE


def dialogue_style_rule(value: Optional[str]) -> str:
    style = normalize_dialogue_style(value)
    return DIALOGUE_STYLE_RULES.get(style, DIALOGUE_STYLE_RULES[DEFAULT_DIALOGUE_STYLE])


def _clean_plot_hooks(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    hooks: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        hooks.append(text)
    return hooks


def _ensure_hook_precision(data: Dict[str, Any]) -> Dict[str, Any]:
    event_schema = data.get("event_schema") if isinstance(data.get("event_schema"), dict) else {}
    hooks = _clean_plot_hooks(event_schema.get("plot_hooks"))
    precise = [h for h in hooks if len(h) >= 8]
    if len(precise) < 6:
        fallback = [
            "钟楼的失踪账本在午夜前会被转移，先去旧邮局拿到钥匙。",
            "北码头的暗号交易将在一小时后开始，先确认接头人的真实身份。",
            "剧院后台的停电并非事故，先找到配电室里被调包的保险丝。",
            "一封写给用户的匿名信提到旧案证人，先去雨巷杂货店调取监控。",
            "城南诊所的病历被篡改，先拿到值班医生留下的备份U盘。",
            "警局档案室今晚临时封存，先借清洁车通行证进入二层走廊。",
            "河堤仓库将在黎明前清空违禁品，先在西门布置一次截获。",
            "咖啡馆二楼的密谈涉及关键名字，先伪装成外送员接近目标桌。",
        ]
        seen = set(precise)
        for item in fallback:
            if item not in seen:
                precise.append(item)
                seen.add(item)
            if len(precise) >= 8:
                break
    event_schema["plot_hooks"] = precise[:10]
    data["event_schema"] = event_schema
    return data


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
                    "- plot_hooks: 6-10 items, each must include 1)明确冲突 2)推进目标 3)下一步可执行行动\n"
                    "- plot_hooks should be actionable now, not vague foreshadowing, no meta commentary\n"
                    "- Each hook should mention concrete people/place/object/time-window when possible\n"
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
        data = _ensure_hook_precision(data)
        scene_setting = str((data.get("scene") or {}).get("setting") or "")
        if _mostly_chinese(scene_setting):
            return data
        retry = _request("IMPORTANT: 必须使用中文作答，不要输出任何英文。")
        if retry:
            retry = _ensure_hook_precision(retry)
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
        "event_schema": {
            "plot_hooks": [
                "钟楼保管室在今夜零点前会转移旧档案，先拿到管理员胸牌进入内门。",
                "港区仓库有一批匿名货即将卸下，先确认接货人的车辆牌照。",
                "雨巷咖啡馆的密谈涉及关键证词，先在二楼靠窗位监听十分钟。",
                "诊所病历被替换成伪造版本，先找到值班护士留下的纸质备份。",
                "旧剧院后台连续停电隐藏了交换行动，先检查配电箱中的异常线缆。",
                "警局封存令将在天亮后生效，先在夜班交接前调出目标档案。",
            ]
        },
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
                    "- mode (string, default \"volcengine\")\n\n"
                    "- dialogue_style (string, one of: natural/balanced/assertive/cinematic/warm/suspense/playful)\n\n"
                    "Rules:\n"
                    "- Keep Chinese only, vivid but practical.\n"
                    "- The NPC should fit the world tone and scene.\n"
                    "- NPC要主动推进剧情，但语气要自然，像日常聊天而不是审问。\n"
                    "- 不要求每轮都提问；可以用陈述、共情、细节描写自然引出plot_hooks。\n"
                    "- 每2-3轮至少推进一条plot_hooks，必要时给出明确下一步行动。\n"
                    "- Avoid repetitive generic lines like “你怎么想/你怎么看/你觉得呢”.\n"
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
        data["dialogue_style"] = normalize_dialogue_style(data.get("dialogue_style"))
        init_sp = str(data.get("init_role_sp") or "")
        if _mostly_chinese(init_sp):
            return data
        retry = _request("IMPORTANT: 必须使用中文作答，不要输出任何英文。")
        if retry:
            retry["dialogue_style"] = normalize_dialogue_style(retry.get("dialogue_style"))
            return retry
    return {
        "mode": "volcengine",
        "dialogue_style": DEFAULT_DIALOGUE_STYLE,
        "role_name": "林一",
        "user_name": "小周",
        "init_role_sp": "简介：你是林一，一位自由插画师。性格开朗，待人温柔体贴。",
        "user_info": "用户叫小周，是你的未婚妻，你通常称呼她为宝贝或周周。",
        "golden_sp": (
            "你使用口语表达，必要时可用括号描述动作和情绪。\n"
            "保持自然聊天感，不要每句都用提问口吻。\n"
            "你仍需主动推进剧情，但可通过日常对话自然带出线索。\n"
            "每2-3轮至少推进一条plot_hooks，用户犹豫时再给明确行动建议。\n"
            "避免重复上一轮问题或内容。"
        ),
        "notes": [],
    }


def generate_npc_reply(context: str, dialogue_style: Optional[str] = None) -> str:
    return generate_npc_reply_with_sp(
        (
            "你是故事中的NPC，请以口语表达，必要时可用括号描述动作或情绪。"
            "你需要在自然聊天里推动剧情，避免闲聊空转或生硬引导。"
            "你必须主动推进剧情，不等待用户决定剧情方向。"
            "优先基于世界里的plot_hooks推进，但不要求每轮都提问。"
            "可以通过陈述、共情、细节描写和轻量建议自然引出线索。"
            "不要表现得像AI。每2-3轮至少出现一次明确推进动作或线索。"
            "避免重复上一轮NPC问题或同义复述；若用户已回答，请确认并给出新进展。"
            "避免连续使用“你决定怎么做”“你会怎么做”这类泛问句。"
            "仅使用中文回答。"
            "回复1-3句，保持节奏感与画面感。"
        ),
        context,
        dialogue_style=dialogue_style,
    )


def generate_npc_reply_with_sp(
    system_prompt: str,
    user_content: str,
    dialogue_style: Optional[str] = None,
) -> str:
    style_rule = dialogue_style_rule(dialogue_style)
    default_rules = (
        "你使用口语进行表达，必要时可用括号描述动作和情绪。"
        "你需要在自然聊天里引导交流，不要表现得太AI。"
        "你要主动推进剧情，不等待用户先决定剧情方向。"
        "不要求每轮提问；可以用陈述句自然带出plot_hooks线索。"
        "每2-3轮至少推进一个剧情点；用户犹豫时给出可执行行动。"
        "避免重复上一轮NPC内容或问句；若用户已回应，请确认并给出新进展。"
        "避免连续使用泛问句，优先自然表达和具体细节。"
        "仅使用中文回答。"
        f"{style_rule}"
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


def generate_npc_reply_stream_with_sp(
    system_prompt: str,
    user_content: str,
    dialogue_style: Optional[str] = None,
):
    style_rule = dialogue_style_rule(dialogue_style)
    if style_rule not in system_prompt:
        system_prompt = f"{system_prompt}\n{style_rule}"
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


def generate_npc_reply_stream(context: str, dialogue_style: Optional[str] = None):
    return generate_npc_reply_stream_with_sp(
        (
            "你是故事中的NPC，请以口语表达，必要时可用括号描述动作或情绪。"
            "你需要在自然聊天里推动剧情发展，避免闲聊空转或泛泛而谈。"
            "你必须主动推进剧情，不等待用户决定剧情走向。"
            "优先围绕plot_hooks推进，但不要求每轮都提问。"
            "不要表现得像AI。可通过陈述、共情、细节自然引出剧情线索。"
            "避免重复上一轮NPC内容或问句；若用户已回应，请确认并给出新进展。"
            "避免连续使用“你怎么想/你怎么看/你觉得呢”等泛问句。"
            "仅使用中文回答。"
            "回复1-3句，保持节奏感与画面感。"
        ),
        context,
        dialogue_style=dialogue_style,
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
