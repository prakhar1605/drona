"""
Redis Caching Layer
Caches LLM-generated question sets to reduce latency and API costs.
Uses Redis Cloud (URL from .env: REDIS_URL).
Falls back gracefully if Redis is unavailable.
"""

import os
import json
import hashlib
import redis
from typing import Optional, List, Dict

_client: Optional[redis.Redis] = None
CACHE_TTL = 60 * 60 * 6  # 6 hours


def _get_client() -> Optional[redis.Redis]:
    global _client
    if _client is not None:
        return _client
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        _client = redis.from_url(url, decode_responses=True, socket_timeout=3)
        _client.ping()
        return _client
    except Exception:
        _client = None
        return None


def _make_key(topics: List[str], difficulty: str, role: str, num_q: int) -> str:
    raw = f"{sorted(topics)}:{difficulty}:{role}:{num_q}"
    return "dronaai:questions:" + hashlib.md5(raw.encode()).hexdigest()


def get_cached_questions(
    topics: List[str], difficulty: str, role: str, num_q: int
) -> Optional[List[Dict]]:
    client = _get_client()
    if not client:
        return None
    try:
        key = _make_key(topics, difficulty, role, num_q)
        data = client.get(key)
        if data:
            return json.loads(data)
    except Exception:
        pass
    return None


def cache_questions(
    topics: List[str], difficulty: str, role: str, num_q: int, questions: List[Dict]
) -> bool:
    client = _get_client()
    if not client:
        return False
    try:
        key = _make_key(topics, difficulty, role, num_q)
        client.setex(key, CACHE_TTL, json.dumps(questions))
        return True
    except Exception:
        return False


def is_redis_connected() -> bool:
    client = _get_client()
    if not client:
        return False
    try:
        client.ping()
        return True
    except Exception:
        return False
