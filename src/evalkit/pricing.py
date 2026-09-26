"""Pricing configuration (docs/TARGET-ARCHITECTURE.md §8.7): explicit, versioned, never guessed.

EvalKit ships **no** prices: provider prices change and cannot be verified from here, so a built-in
table would be a claim we cannot back. The operator supplies a table (`--pricing FILE`, or
`policy.pricing`) with a `version` label of their choosing; it is frozen into the run's policy,
and every attempt records the version its cost came from, so a later price change never rewrites
history.

What an estimate can and cannot say:

* `usd` is an **estimate**: tokens reported by the provider x the table's price. It is exact only
  to the extent the table is.
* A model with no entry is *unpriced* (`usd is None`), and a call whose provider reported no
  token counts is *no usage* (`usd is None`). Neither is ever zero.
* Cached responses cost nothing in the run that reused them (that is recorded by the caller as
  0.0, not computed here).
"""

from __future__ import annotations

import json
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.errors import ConfigError
from evalkit.hashing import stable_hash

PRICING_DOMAIN = "evalkit-pricing-v1"
MAX_PRICE_MODELS = 2000
_MODEL_KEYS = {"provider", "model", "input_per_mtok", "output_per_mtok"}


@dataclass(frozen=True)
class Cost:
    usd: float | None  # None: unknown (see `reason`)
    version: str | None = None  # the price table it was computed from
    reason: str | None = None  # "unpriced" | "no_usage" when usd is None


@dataclass(frozen=True)
class ModelPrice:
    provider: str
    model: str
    input_per_mtok: float  # USD per million input tokens
    output_per_mtok: float


def _price(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ConfigError(f"{where} must be a finite, non-negative number")
    return float(value)


@dataclass(frozen=True)
class PriceTable:
    version: str
    models: Mapping[tuple[str, str], ModelPrice] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> PriceTable:
        if not isinstance(m, Mapping):
            raise ConfigError("pricing must be a table with `version` and `models`")
        unknown = set(m) - {"version", "currency", "models", "note"}
        if unknown:
            raise ConfigError(f"pricing: unknown key(s) {sorted(unknown)}")
        version = m.get("version")
        if not isinstance(version, str) or not version.strip() or len(version) > 64:
            raise ConfigError("pricing needs an explicit `version` label (1-64 characters)")
        if m.get("currency", "USD") != "USD":
            raise ConfigError("pricing: only USD is supported")
        raw = m.get("models", [])
        if not isinstance(raw, list | tuple) or len(raw) > MAX_PRICE_MODELS:
            raise ConfigError(f"pricing.models must be a list of at most {MAX_PRICE_MODELS}")
        models: dict[tuple[str, str], ModelPrice] = {}
        for i, e in enumerate(raw):
            where = f"pricing.models[{i}]"
            if not isinstance(e, Mapping) or set(e) != _MODEL_KEYS:
                raise ConfigError(f"{where} needs exactly {sorted(_MODEL_KEYS)}")
            for key in ("provider", "model"):
                if not isinstance(e[key], str) or not e[key]:
                    raise ConfigError(f"{where}.{key} must be a non-empty string")
            price = ModelPrice(
                e["provider"],
                e["model"],
                _price(e["input_per_mtok"], f"{where}.input_per_mtok"),
                _price(e["output_per_mtok"], f"{where}.output_per_mtok"),
            )
            if (price.provider, price.model) in models:
                raise ConfigError(f"{where}: {price.provider}/{price.model} is listed twice")
            models[(price.provider, price.model)] = price
        return cls(version, models)

    @classmethod
    def from_file(cls, path: str | Path) -> PriceTable:
        p = Path(path)
        try:
            data = p.read_bytes()
            text = data.decode("utf-8")
            parsed = tomllib.loads(text) if p.suffix == ".toml" else json.loads(text)
        except OSError as e:
            raise ConfigError(f"cannot read pricing file {path}: {e}") from e
        except (ValueError, UnicodeDecodeError) as e:
            raise ConfigError(f"cannot parse pricing file {path}: {e}") from e
        return cls.from_mapping(parsed)

    def to_mapping(self) -> dict[str, Any]:
        """Plain JSON, models sorted: what is stored in a run's policy."""
        return {
            "version": self.version,
            "currency": "USD",
            "models": [
                {
                    "provider": p.provider,
                    "model": p.model,
                    "input_per_mtok": p.input_per_mtok,
                    "output_per_mtok": p.output_per_mtok,
                }
                for _, p in sorted(self.models.items())
            ],
        }

    @property
    def content_hash(self) -> str:
        return stable_hash(PRICING_DOMAIN, self.to_mapping())

    def price_of(self, provider: str | None, model: str | None) -> ModelPrice | None:
        if provider is None or model is None:
            return None
        return self.models.get((provider, model))

    def cost(
        self,
        provider: str | None,
        model: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> Cost:
        price = self.price_of(provider, model)
        if price is None:
            return Cost(None, None, "unpriced")
        if input_tokens is None or output_tokens is None:
            return Cost(None, None, "no_usage")
        usd = (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1e6
        return Cost(usd, self.version)

    def upper_bound(
        self, provider: str | None, model: str | None, input_tokens: int, output_tokens: int
    ) -> float | None:
        """The most a call of at most these sizes can cost, or None when the model is unpriced."""
        price = self.price_of(provider, model)
        if price is None:
            return None
        return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1e6


NO_PRICING = PriceTable("none")
