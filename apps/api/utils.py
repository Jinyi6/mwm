from __future__ import annotations

import json
import uuid
import random
import string
from datetime import datetime
from typing import Any, Dict


def new_id() -> str:
    return uuid.uuid4().hex


def new_session_id() -> str:
    letters = "".join(random.choices(string.ascii_uppercase, k=4))
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{letters}{ts}"


def dumps_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def loads_json(data: str, default: Any) -> Any:
    try:
        return json.loads(data)
    except Exception:
        return default


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))
