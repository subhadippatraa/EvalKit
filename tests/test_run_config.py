"""RunConfig, EvaluatorSpec, TargetSpec: frozen, strict, hashable; identity vs execution hashes."""

import hashlib
import json

import pytest
from conftest import EM, JUDGE, case, make_run, spec
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from evalkit import EvaluatorSpec, RunConfig, RunError, TargetSpec
from evalkit.runs import SCORING_VERSION

H = "a" * 64


def cfg(*evaluators, **kw):
    return RunConfig(evaluators=list(evaluators), **kw)


# --- evaluator identity ----------------------------------------------------------------------


def test_evaluator_keys_are_pinned():
    """The key format is stored in every run; changing it silently would orphan old runs."""
    assert EM.key == "exact_match:em:45b5b2533aa4"
    assert spec("regex", "r1", pattern="^a").key.startswith("regex:r1:")


def test_a_key_changes_with_anything_that_can_change_a_score():
    base = spec("llm_judge", "q", model="m1", temperature=0.0)
    assert spec("llm_judge", "q", model="m2", temperature=0.0).key != base.key  # params
    assert spec("llm_judge", "q", model="m1", temperature=0.5).key != base.key
    assert spec("llm_judge", "q2", model="m1", temperature=0.0).key != base.key  # name
    assert spec("regex", "q", model="m1", temperature=0.0).key != base.key  # kind
    v2 = EvaluatorSpec(kind="llm_judge", name="q", version=2, params=base.params)
    assert v2.key != base.key  # the evaluator's own scoring logic changed


def test_equal_specs_have_equal_keys_regardless_of_param_order():
    a = EvaluatorSpec(kind="regex", name="r", params={"x": 1, "y": [1, 2]})
    b = EvaluatorSpec(kind="regex", name="r", params={"y": [1, 2], "x": 1})
    assert a.key == b.key


def test_param_lists_are_ordered_data_and_1_differs_from_1_point_0():
    assert spec(k=[1, 2]).key != spec(k=[2, 1]).key
    assert spec(k=1).key != spec(k=1.0).key  # the hash is a function of exactly what was given


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "Bad Kind", "name": "n"},
        {"kind": "k", "name": "has:colon"},  # ':' separates the parts of a key
        {"kind": "k", "name": ""},
        {"kind": "k", "name": "n", "version": 0},
        {"kind": "k", "name": "n", "params": {"x": float("nan")}},
        {"kind": "k", "name": "n", "params": {"x": float("inf")}},
        {"kind": "k", "name": "n", "params": {1: "non-string key"}},
        {"kind": "k", "name": "n", "params": {"x": object()}},
        {"kind": "k", "name": "n", "params": {"x": "\ud800"}},
        {"kind": "k", "name": "n", "surprise": 1},
    ],
)
def test_invalid_evaluator_specs_are_refused(bad):
    with pytest.raises(ValidationError):
        EvaluatorSpec(**bad)


def test_strict_types_are_not_coerced():
    with pytest.raises(ValidationError):
        EvaluatorSpec(kind="k", name="n", version="2")
    with pytest.raises(ValidationError):
        EvaluatorSpec(kind="k", name="n", version=True)
    with pytest.raises(ValidationError):
        RunConfig(scoring_version="1")


# --- config ----------------------------------------------------------------------------------


def test_defaults_describe_a_precomputed_run_with_current_scoring():
    c = RunConfig()
    assert (c.target.kind, c.evaluators, c.policy, c.scoring_version) == (
        "precomputed", (), {}, SCORING_VERSION,
    )  # fmt: skip


def test_a_config_is_frozen():
    c = cfg(EM)
    with pytest.raises(ValidationError):
        c.policy = {"x": 1}
    with pytest.raises(ValidationError):
        c.evaluators = ()
    assert isinstance(c.evaluators, tuple)  # a list was accepted, an immutable tuple is stored
    with pytest.raises(ValidationError):
        EM.name = "other"


def test_config_accepts_plain_dicts_and_round_trips_through_json():
    c = RunConfig.model_validate(
        {
            "target": {"kind": "model", "identity": {"model": "m", "prompt_sha": "abc"}},
            "evaluators": [{"kind": "exact_match", "name": "em"}],
            "policy": {"concurrency": 8},
        }
    )
    again = RunConfig.model_validate_json(c.model_dump_json())
    assert again == c and again.evaluator_keys == c.evaluator_keys
    assert again.identity_hash(H) == c.identity_hash(H)


@pytest.mark.parametrize(
    "bad",
    [
        {"target": {"kind": "telepathy"}},
        {"target": {"kind": "model", "identity": {"x": float("nan")}}},
        {"policy": {"x": float("inf")}},
        {"policy": {"nested": {"deep": object()}}},
        {"scoring_version": 0},
        {"unknown": 1},
    ],
)
def test_invalid_configs_are_refused(bad):
    with pytest.raises(ValidationError):
        RunConfig.model_validate(bad)


def test_duplicate_evaluators_and_too_many_are_refused():
    with pytest.raises(ValidationError, match="same key"):
        cfg(EM, spec())
    with pytest.raises(ValidationError, match="at most"):
        cfg(*[spec("k", f"n{i}") for i in range(33)])
    cfg(*[spec("k", f"n{i}") for i in range(32)])  # exactly the limit is fine


def test_two_evaluators_of_one_kind_with_different_params_coexist():
    c = cfg(spec("regex", "a", pattern="x"), spec("regex", "b", pattern="y"))
    assert len(c.evaluator_keys) == 2


# --- identity hash vs exec hash --------------------------------------------------------------


def test_hashes_are_pinned_and_follow_the_documented_recipe():
    """Stored in every run: a silent change would make old and new runs look non-comparable.
    The recipe is also re-derived here by hand from the design (section 3.3)."""
    c = cfg(EM, JUDGE)
    assert c.identity_hash(H) == "69151cc66dca3fcd5ae9221f872e59c4f138d408ab1c7a17ac1479b3df674fc8"
    assert cfg(policy={"c": 1}).exec_hash == (
        "ba8c4a465cf5cba08ca204587e77d1c52bab0d32201bdbe51e370d02c32244f5"
    )
    payload = {
        "dataset": H,
        "target": {"kind": "precomputed", "identity": {}},
        "evaluators": sorted([EM.key, JUDGE.key]),
        "scoring_version": SCORING_VERSION,
    }
    by_hand = hashlib.sha256(
        b"evalkit-run-identity-v1\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert c.identity_hash(H) == by_hand


def test_identity_ignores_evaluator_order_and_execution_policy():
    a = cfg(EM, JUDGE, policy={"concurrency": 1})
    b = cfg(JUDGE, EM, policy={"concurrency": 64, "retries": 5})
    assert a.identity_hash(H) == b.identity_hash(H)
    assert a.exec_hash != b.exec_hash


def test_exec_hash_depends_only_on_policy():
    assert cfg(EM, policy={"c": 1}).exec_hash == cfg(JUDGE, policy={"c": 1}).exec_hash
    assert cfg(EM, policy={"c": 1}).exec_hash != cfg(EM, policy={"c": 2}).exec_hash
    assert cfg(EM, policy={"a": 1, "b": 2}).exec_hash == cfg(EM, policy={"b": 2, "a": 1}).exec_hash


@pytest.mark.parametrize(
    "other",
    [
        cfg(EM),  # fewer evaluators
        cfg(EM, spec("llm_judge", "quality", model="judge-2", rubric="r1")),  # judge changed
        cfg(EM, JUDGE, target=TargetSpec(kind="model", identity={"model": "m"})),
        RunConfig(evaluators=[EM, JUDGE], scoring_version=SCORING_VERSION + 1),
    ],
)
def test_anything_that_affects_scores_changes_the_identity(other):
    assert other.identity_hash(H) != cfg(EM, JUDGE).identity_hash(H)


def test_identity_depends_on_dataset_content_and_the_target_identity():
    c = cfg(EM)
    assert c.identity_hash("a" * 64) != c.identity_hash("b" * 64)
    t1 = RunConfig(target=TargetSpec(kind="model", identity={"model": "m1"}))
    t2 = RunConfig(target=TargetSpec(kind="model", identity={"model": "m2"}))
    assert t1.identity_hash(H) != t2.identity_hash(H)


def test_identity_domain_separation_from_exec_hash():
    """An empty policy and an otherwise-empty identity must not collide."""
    c = RunConfig()
    assert c.identity_hash(H) != c.exec_hash


scalars = st.one_of(
    st.none(), st.booleans(), st.integers(-10**6, 10**6), st.text(max_size=8),
    st.floats(allow_nan=False, allow_infinity=False),
)  # fmt: skip
json_dicts = st.dictionaries(st.text(min_size=1, max_size=6), scalars, max_size=5)


@settings(max_examples=100, deadline=None)
@given(params=json_dicts, policy=json_dicts, perm=st.randoms())
def test_property_identity_is_order_and_policy_independent(params, policy, perm):
    evaluators = [spec("k", f"n{i}", **params, extra=i) for i in range(4)]
    shuffled = evaluators[:]
    perm.shuffle(shuffled)
    a = RunConfig(evaluators=evaluators)
    b = RunConfig(evaluators=shuffled, policy=policy)
    assert a.identity_hash(H) == b.identity_hash(H)
    assert RunConfig.model_validate_json(b.model_dump_json()).exec_hash == b.exec_hash


@settings(max_examples=100, deadline=None)
@given(a=json_dicts, b=json_dicts)
def test_property_distinct_params_give_distinct_keys(a, b):
    ka, kb = spec("k", "n", **a).key, spec("k", "n", **b).key
    # equal keys only for params that are equal as JSON (1 == 1.0 in Python, not in the hash)
    if ka == kb:
        assert EvaluatorSpec(kind="k", name="n", params=a).params == (
            EvaluatorSpec(kind="k", name="n", params=b).params
        )


# --- through the service ---------------------------------------------------------------------


def test_scoring_version_other_than_current_is_refused_at_creation(kit):
    kit.datasets.import_cases("qa", [case("a")])
    with pytest.raises(RunError, match="scoring_version"):
        kit.runs.create("qa", RunConfig(scoring_version=SCORING_VERSION + 1))


def test_an_invalid_config_mapping_raises_run_error_with_every_problem(kit):
    kit.datasets.import_cases("qa", [case("a")])
    with pytest.raises(RunError, match="invalid run config") as e:
        kit.runs.create("qa", {"evaluators": [{"kind": "K", "name": "n"}], "bogus": 1})
    assert "bogus" in str(e.value) and "evaluators" in str(e.value)


def test_an_oversized_config_is_refused(kit):
    kit.datasets.import_cases("qa", [case("a")])
    with pytest.raises(RunError, match="larger than"):
        kit.runs.create("qa", RunConfig(policy={"blob": "x" * (300 * 1024)}))


def test_the_recorded_config_is_exactly_what_was_frozen(kit):
    config = RunConfig(evaluators=[EM, JUDGE], policy={"concurrency": 4})
    run = make_run(kit, config=config)
    assert run.config == config
    assert kit.runs.get(run.id).config.evaluator_keys == sorted([EM.key, JUDGE.key])
