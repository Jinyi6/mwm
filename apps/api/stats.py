from __future__ import annotations

import math
from typing import Dict, List, Tuple
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


def _trend_summary(desires: List[int], window: int = 6) -> Dict[str, object]:
    if len(desires) < 2:
        return {
            "direction": "flat",
            "delta": 0.0,
            "summary": "数据不足，暂无趋势判断",
            "window": min(window, len(desires)),
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
    emotion_distribution = {label: 0 for label in EMOTION_LABELS}
    for m in items:
        if m.emotion_label in emotion_distribution:
            emotion_distribution[m.emotion_label] += 1
    texts = [m.content for m in items]
    topics = [m.topic_tag or "" for m in items]
    avg_turn_duration = _avg_turn_duration(items)
    trend = _trend_summary(desires, window=6)
    return {
        "total_messages": total_messages,
        "total_turns": turns,
        "avg_desire": round(avg_desire, 2),
        "emotion_distribution": emotion_distribution,
        "lexical_diversity": lexical_diversity(texts),
        "topic_diversity": topic_diversity(topics),
        "desire_series": desires,
        "avg_turn_duration_sec": round(avg_turn_duration, 2),
        "trend_summary": trend["summary"],
        "trend_direction": trend["direction"],
        "trend_delta": trend["delta"],
        "trend_window": trend["window"],
    }
