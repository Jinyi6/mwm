from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any
from sqlmodel import Session, select

from .models import Message
from .memory import EMOTION_LABELS


def _tokenize(text: str) -> List[str]:
    return [t for t in text.replace("\n", " ").split(" ") if t]


def lexical_diversity(texts: List[str]) -> Dict[str, float]:
    tokens = []
    for t in texts:
        tokens.extend(_tokenize(t))
    if not tokens:
        return {"ttr": 0.0, "mtld": 0.0}
    unique = len(set(tokens))
    ttr = unique / max(1, len(tokens))
    mtld = _mtld(tokens)
    return {"ttr": round(ttr, 4), "mtld": round(mtld, 4)}


def _mtld(tokens: List[str], threshold: float = 0.72) -> float:
    if not tokens:
        return 0.0
    factors = 0
    types = set()
    count = 0
    for tok in tokens:
        count += 1
        types.add(tok)
        ttr = len(types) / count
        if ttr <= threshold:
            factors += 1
            types = set()
            count = 0
    if count > 0:
        factors += (1 - (len(types) / max(1, count))) / (1 - threshold)
    if factors == 0:
        return float(len(tokens))
    return len(tokens) / factors


def topic_diversity(topics: List[str]) -> Dict[str, float]:
    total = len([t for t in topics if t])
    if total == 0:
        return {"unique": 0, "entropy": 0.0}
    counts: Dict[str, int] = {}
    for t in topics:
        if not t:
            continue
        counts[t] = counts.get(t, 0) + 1
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p + 1e-9, 2)
    return {"unique": len(counts), "entropy": round(entropy, 4)}


def _avg_turn_duration(items: List[Message]) -> float:
    if not items:
        return 0.0
    spans: Dict[int, Tuple[float, float]] = {}
    for m in items:
        ts = m.created_at.timestamp()
        if m.turn_index not in spans:
            spans[m.turn_index] = (ts, ts)
        else:
            start, end = spans[m.turn_index]
            spans[m.turn_index] = (min(start, ts), max(end, ts))
    durations = [end - start for start, end in spans.values()]
    if not durations:
        return 0.0
    return sum(durations) / len(durations)


def _turn_durations(items: List[Message]) -> List[float]:
    if not items:
        return []
    spans: Dict[int, Tuple[float, float]] = {}
    for m in items:
        ts = m.created_at.timestamp()
        if m.turn_index not in spans:
            spans[m.turn_index] = (ts, ts)
        else:
            start, end = spans[m.turn_index]
            spans[m.turn_index] = (min(start, ts), max(end, ts))
    return [round(end - start, 2) for _, (start, end) in sorted(spans.items())]


def _response_times(items: List[Message]) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
    if not items:
        return [], []
    by_turn: Dict[int, List[Message]] = {}
    for m in items:
        by_turn.setdefault(m.turn_index, []).append(m)
    npc_times: List[float] = []
    user_times: List[float] = []
    per_turn: List[Dict[str, Any]] = []
    for _, msgs in sorted(by_turn.items()):
        msgs = sorted(msgs, key=lambda m: m.created_at)
        if len(msgs) < 2:
            continue
        first, second = msgs[0], msgs[1]
        delta = (second.created_at - first.created_at).total_seconds()
        if first.role == "user" and second.role == "npc":
            npc_times.append(delta)
            per_turn.append(
                {"turn_index": first.turn_index, "direction": "user_to_npc", "response_sec": round(delta, 2)}
            )
        elif first.role == "npc" and second.role == "user":
            user_times.append(delta)
            per_turn.append(
                {"turn_index": first.turn_index, "direction": "npc_to_user", "response_sec": round(delta, 2)}
            )
    return npc_times, user_times, per_turn


def _trend_summary(desires: List[int], window: int = 6) -> Dict[str, object]:
    if len(desires) == 0:
        return {
            "direction": "flat",
            "delta": 0.0,
            "summary": "暂无趋势数据",
            "window": 0,
        }
    if len(desires) == 1:
        return {
            "direction": "flat",
            "delta": 0.0,
            "summary": "仅1轮数据，趋势暂不判断（+0.0）",
            "window": 1,
        }
    recent = desires[-window:] if window > 1 else desires[-1:]
    delta = float(recent[-1] - recent[0]) if len(recent) >= 2 else 0.0
    if abs(delta) < 0.5:
        direction = "flat"
        summary = f"近{len(recent)}轮欲望整体稳定（{delta:+.1f}）"
    elif delta > 0:
        direction = "up"
        summary = f"近{len(recent)}轮欲望整体上升（{delta:+.1f}）"
    else:
        direction = "down"
        summary = f"近{len(recent)}轮欲望整体下降（{delta:+.1f}）"
    return {
        "direction": direction,
        "delta": round(delta, 2),
        "summary": summary,
        "window": len(recent),
    }


def _emotion_distribution(items: List[Message], role: str | None = None) -> Dict[str, int]:
    distribution = {label: 0 for label in EMOTION_LABELS}
    for m in items:
        if role and m.role != role:
            continue
        if not m.emotion_label:
            continue
        if m.emotion_label in distribution:
            distribution[m.emotion_label] += 1
        else:
            distribution["other"] += 1
    return distribution


def compute_stats(session: Session, chat_id: str) -> Dict[str, any]:
    stmt = (
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.turn_index, Message.created_at)
    )
    items = session.exec(stmt).all()
    total_messages = len(items)
    turns = len(set([m.turn_index for m in items])) if items else 0
    desires = [m.desire_score for m in items if m.desire_score is not None]
    avg_desire = sum(desires) / len(desires) if desires else 0.0
    emotion_all = _emotion_distribution(items)
    emotion_user = _emotion_distribution(items, role="user")
    emotion_npc = _emotion_distribution(items, role="npc")
    # Keep backward-compatible field but fix semantics:
    # emotion_distribution defaults to user side (aligned with desire metrics).
    emotion_distribution = emotion_user if sum(emotion_user.values()) > 0 else emotion_all
    texts = [m.content for m in items]
    topics = [m.topic_tag or "" for m in items]
    avg_turn_duration = _avg_turn_duration(items)
    per_turn_durations = _turn_durations(items)
    npc_times, user_times, per_turn_responses = _response_times(items)
    avg_npc_response = sum(npc_times) / len(npc_times) if npc_times else 0.0
    avg_user_response = sum(user_times) / len(user_times) if user_times else 0.0
    trend = _trend_summary(desires, window=6)
    return {
        "total_messages": total_messages,
        "total_turns": turns,
        "avg_desire": round(avg_desire, 2),
        "emotion_distribution": emotion_distribution,
        "emotion_distribution_all": emotion_all,
        "emotion_distribution_user": emotion_user,
        "emotion_distribution_npc": emotion_npc,
        "lexical_diversity": lexical_diversity(texts),
        "topic_diversity": topic_diversity(topics),
        "desire_series": desires,
        "per_turn_durations": per_turn_durations,
        "avg_turn_duration_sec": round(avg_turn_duration, 2),
        "avg_npc_response_sec": round(avg_npc_response, 2),
        "avg_user_response_sec": round(avg_user_response, 2),
        "per_turn_responses": per_turn_responses,
        "trend_summary": trend["summary"],
        "trend_direction": trend["direction"],
        "trend_delta": trend["delta"],
        "trend_window": trend["window"],
    }
