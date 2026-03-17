from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def loads_json(value: str, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def normalize_tag(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip().lower())
    cleaned = cleaned.replace("_", " ")
    return cleaned


def normalize_tags(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    tags: list[str] = []
    for value in values:
        tag = normalize_tag(value)
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_seed(base: int, offset: int) -> int:
    random.seed(base + offset)
    return random.randint(1, 2_147_483_647)
