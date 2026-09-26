"""Pricing configuration: explicit, versioned, and honest about what it does not know."""

import math

import pytest

from evalkit.errors import ConfigError
from evalkit.pricing import NO_PRICING, PriceTable


def table(**kw):
    return PriceTable.from_mapping(
        {
            "version": "2026-09-example",
            "models": [
                {"provider": "p", "model": "m", "input_per_mtok": 3.0, "output_per_mtok": 15.0}
            ],
        }
        | kw
    )


def test_a_priced_call_costs_tokens_times_price_per_million():
    cost = table().cost("p", "m", 1_000_000, 2_000_000)
    assert cost.usd == pytest.approx(3.0 + 30.0) and cost.version == "2026-09-example"


def test_unknown_model_or_provider_is_unpriced_never_zero():
    t = table()
    for provider, model in (("p", "other"), ("other", "m"), (None, None)):
        c = t.cost(provider, model, 10, 10)
        assert c.usd is None and c.version is None and c.reason == "unpriced"


def test_unknown_usage_is_unknown_never_zero():
    t = table()
    for tokens in ((None, 5), (5, None), (None, None)):
        c = t.cost("p", "m", *tokens)
        assert c.usd is None and c.reason == "no_usage"


def test_the_empty_default_table_prices_nothing():
    assert NO_PRICING.cost("p", "m", 1, 1).usd is None and NO_PRICING.version == "none"
    assert not NO_PRICING.models


def test_a_table_needs_an_explicit_version_and_valid_prices():
    for bad in (
        {"models": []},
        {"version": "", "models": []},
        {"version": "v", "models": [{"provider": "p", "model": "m", "input_per_mtok": 1}]},
        {"version": "v", "models": [{"provider": "p", "model": "m", "input_per_mtok": -1,
                                     "output_per_mtok": 1}]},
        {"version": "v", "models": [{"provider": "p", "model": "m", "input_per_mtok": math.inf,
                                     "output_per_mtok": 1}]},
        {"version": "v", "models": [{"provider": "p", "model": "m", "input_per_mtok": True,
                                     "output_per_mtok": 1}]},
        {"version": "v", "models": [], "surprise": 1},
        {"version": "v", "currency": "EUR", "models": []},
    ):  # fmt: skip
        with pytest.raises(ConfigError):
            PriceTable.from_mapping(bad)


def test_a_duplicate_model_entry_is_refused():
    entry = {"provider": "p", "model": "m", "input_per_mtok": 1, "output_per_mtok": 1}
    with pytest.raises(ConfigError, match="twice"):
        PriceTable.from_mapping({"version": "v", "models": [entry, entry]})


def test_the_content_hash_names_the_prices_and_ignores_order():
    a = PriceTable.from_mapping(
        {
            "version": "v",
            "models": [
                {"provider": "a", "model": "1", "input_per_mtok": 1, "output_per_mtok": 2},
                {"provider": "b", "model": "2", "input_per_mtok": 3, "output_per_mtok": 4},
            ],
        }
    )
    b = PriceTable.from_mapping(
        {"version": "v", "models": list(reversed(a.to_mapping()["models"]))}
    )
    c = PriceTable.from_mapping({"version": "v", "models": [*a.to_mapping()["models"][:1]]})
    assert a.content_hash == b.content_hash != c.content_hash


def test_the_mapping_round_trips():
    t = table()
    assert PriceTable.from_mapping(t.to_mapping()) == t


def test_a_pricing_file_is_loaded_from_toml_or_json(tmp_path):
    toml = tmp_path / "p.toml"
    toml.write_text(
        'version = "v1"\n[[models]]\nprovider = "p"\nmodel = "m"\n'
        "input_per_mtok = 1.0\noutput_per_mtok = 2.0\n"
    )
    js = tmp_path / "p.json"
    js.write_text(
        '{"version":"v1","models":[{"provider":"p","model":"m",'
        '"input_per_mtok":1.0,"output_per_mtok":2.0}]}'
    )
    assert PriceTable.from_file(toml) == PriceTable.from_file(js)
    with pytest.raises(ConfigError, match="cannot read"):
        PriceTable.from_file(tmp_path / "missing.toml")


def test_upper_bound_cost_uses_the_price_or_says_it_cannot():
    t = table()
    assert t.upper_bound("p", "m", 1_000_000, 1_000_000) == pytest.approx(18.0)
    assert t.upper_bound("p", "x", 1, 1) is None
