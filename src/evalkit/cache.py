"""The provider-response cache (docs/TARGET-ARCHITECTURE.md §8.6): a content-addressed store of
*validated* provider responses, so an exact repeat of a request is never billed twice.

Identity. The key is a hash of **every field of the `LLMRequest`** (system, user, tool name /
description / schema, temperature, max_tokens, sample_index, role; a field added to the request
later joins the key automatically -- only `timeout_s`, which cannot change an answer, is ignored),
plus the provider, model, endpoint (region / host) and a *scope*: the evaluator key (which covers
rubric, prompt version and scoring version) for a judge call, the target's identity for a target
call. Two configurations that differ in any of these can never share an entry.

Rules.
* Only responses that passed validation are written (the caller writes after `validate`), so a
  malformed answer is never replayed. Failures are never cached.
* Only deterministic settings are cached: `temperature <= max_temperature` (default 0). Judge
  calls are eligible; target calls only when `targets` is set (a target's nondeterminism is the
  thing being measured, and caching it would hide a regression).
* `replay` mode reads only and never calls the provider on a miss: exact reproduction or nothing.
* Raw provider payloads kept as failure evidence are not cached.
* Identical requests in flight at once are serialized per key, so the second waits and hits
  instead of paying for the same answer twice.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Literal

from evalkit.hashing import stable_hash
from evalkit.llm import LLMRequest, LLMResponse, Usage

CACHE_DOMAIN = "evalkit-llm-cache-v1"
CACHE_FORMAT = 1
KEY_IGNORES = frozenset({"timeout_s"})  # request fields that cannot change an answer


@dataclass(frozen=True)
class CachePolicy:
    mode: Literal["off", "readwrite", "replay"] = "off"
    targets: bool = False  # also cache model-target calls (a controlled replay only)
    max_temperature: float = 0.0  # calls above this are non-deterministic and never cached

    def __post_init__(self) -> None:
        if self.mode not in ("off", "readwrite", "replay"):
            raise ValueError("cache.mode must be 'off', 'readwrite' or 'replay'")
        if not isinstance(self.targets, bool):
            raise ValueError("cache.targets must be true or false")
        t = self.max_temperature
        if isinstance(t, bool) or not isinstance(t, int | float) or not 0 <= t <= 2:
            raise ValueError("cache.max_temperature must be a number in [0, 2]")

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> CachePolicy:
        unknown = set(m) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown cache setting(s): {sorted(unknown)}")
        return cls(**m)


def cache_key(
    req: LLMRequest, *, provider: str | None, model: str | None, endpoint: str | None, scope: str
) -> str:
    fields = {k: v for k, v in asdict(req).items() if k not in KEY_IGNORES}
    return stable_hash(
        CACHE_DOMAIN,
        {
            "format": CACHE_FORMAT,
            "provider": provider,
            "model": model,
            "endpoint": endpoint,
            "scope": scope,
            "request": fields,
        },
    )


def encode_response(resp: LLMResponse) -> str:
    return json.dumps(
        {
            "v": CACHE_FORMAT,
            "payload": resp.payload,
            "text": resp.text,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "request_id": resp.request_id,
            "provider_latency_ms": resp.provider_latency_ms,
            "stop_reason": resp.stop_reason,
            "provider_stop": resp.provider_stop,
        },
        allow_nan=False,
    )


def decode_response(text: str) -> LLMResponse | None:
    """The cached response, or None if the entry is unreadable (treated as a miss)."""
    try:
        d = json.loads(text)
        if not isinstance(d, dict) or d.get("v") != CACHE_FORMAT:
            return None
        if not isinstance(d["stop_reason"], str):
            return None
        return LLMResponse(
            payload=d["payload"],
            text=d["text"],
            usage=Usage(d["input_tokens"], d["output_tokens"]),
            request_id=d["request_id"],
            provider_latency_ms=d["provider_latency_ms"],
            stop_reason=d["stop_reason"],
            provider_stop=d["provider_stop"],
        )
    except (ValueError, KeyError, TypeError):
        return None


class ResponseCache:
    """A policy over a store (`cache_get` / `cache_put`), plus the per-key lock."""

    def __init__(self, store: Any, policy: CachePolicy):
        self.store = store
        self.policy = policy
        self._mutex = threading.Lock()
        self._locks: dict[str, list[Any]] = {}  # key -> [lock, users]

    @property
    def active(self) -> bool:
        return self.policy.mode != "off"

    @property
    def replay(self) -> bool:
        return self.policy.mode == "replay"

    def eligible(self, req: LLMRequest) -> bool:
        if not self.active or req.temperature > self.policy.max_temperature:
            return False
        return req.role == "evaluator" or self.policy.targets

    def get(self, key: str) -> LLMResponse | None:
        text = self.store.cache_get(key)
        if text is None:
            return None
        resp = decode_response(text)
        if resp is None:  # unreadable: drop it so the next validated success can replace it
            self.store.cache_delete(key)
        return resp

    def put(self, key: str, provider: str | None, model: str | None, resp: LLMResponse) -> None:
        if self.policy.mode == "readwrite":
            self.store.cache_put(key, provider or "", model or "", encode_response(resp))

    @contextmanager
    def key_lock(self, key: str) -> Iterator[None]:
        with self._mutex:
            entry = self._locks.setdefault(key, [threading.Lock(), 0])
            entry[1] += 1
        entry[0].acquire()
        try:
            yield
        finally:
            entry[0].release()
            with self._mutex:
                entry[1] -= 1
                if entry[1] == 0:
                    del self._locks[key]
