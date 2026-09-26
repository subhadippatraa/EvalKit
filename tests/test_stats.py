"""Statistics: known values, edge cases, seeded determinism, properties."""

import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evalkit import stats

floats = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)


@pytest.mark.parametrize("fn", [stats.mean, stats.stdev, lambda x: stats.quantile(x, 0.5)])
def test_no_values_and_non_finite_values_are_refused(fn):
    with pytest.raises(ValueError):
        fn([])
    for bad in (float("nan"), float("inf"), True, "1"):
        with pytest.raises(ValueError, match="finite"):
            fn([1.0, bad])


def test_mean_stdev_quantile_known_values():
    xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert stats.mean(xs) == 5.0
    assert stats.stdev(xs) == pytest.approx(math.sqrt(32 / 7))
    assert stats.stdev([3.0]) == 0.0
    assert stats.median([1, 2, 3, 4]) == 2.5
    assert stats.quantile([1, 2, 3, 4, 5], 0.0) == 1 and stats.quantile([1, 2, 3, 4, 5], 1.0) == 5
    assert stats.quantile([10, 20], 0.25) == 12.5
    assert stats.quantile([7], 0.95) == 7
    with pytest.raises(ValueError):
        stats.quantile([1], 1.5)


def test_fsum_makes_the_mean_independent_of_order():
    xs = [0.1] * 10 + [1e16, -1e16]
    assert stats.mean(xs) == stats.mean(list(reversed(xs)))


@pytest.mark.parametrize(
    "k,n,low,high",
    [(0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0), (5, 10, 0.2366, 0.7634), (1, 1, 0.2065, 1.0)],
)
def test_wilson_matches_published_values(k, n, low, high):
    lo, hi = stats.wilson(k, n)
    assert lo == pytest.approx(low, abs=1e-3) and hi == pytest.approx(high, abs=1e-3)


@pytest.mark.parametrize("k,n", [(-1, 5), (6, 5), (0, 0), (1, -1)])
def test_wilson_rejects_impossible_counts(k, n):
    with pytest.raises(ValueError):
        stats.wilson(k, n)


@given(n=st.integers(1, 5000), data=st.data())
def test_property_wilson_is_a_valid_interval_containing_the_estimate(n, data):
    k = data.draw(st.integers(0, n))
    lo, hi = stats.wilson(k, n)
    assert 0.0 <= lo <= k / n <= hi <= 1.0 + 1e-12


def test_wilson_narrows_with_more_data():
    widths = [stats.wilson(n // 2, n)[1] - stats.wilson(n // 2, n)[0] for n in (10, 100, 1000)]
    assert widths[0] > widths[1] > widths[2]


def test_bootstrap_is_seeded_deterministic_and_seed_sensitive():
    rng = random.Random(1)
    xs = [rng.gauss(0, 1) for _ in range(40)]
    a = stats.bootstrap_ci(xs, seed=7, b=300)
    assert a == stats.bootstrap_ci(xs, seed=7, b=300)
    assert a != stats.bootstrap_ci(xs, seed=8, b=300)
    lo, hi = a
    assert lo <= stats.mean(xs) <= hi


def test_bootstrap_of_a_constant_is_that_constant_and_a_single_value_has_no_interval():
    assert stats.bootstrap_ci([3.0] * 10, seed=1, b=50) == (3.0, 3.0)
    assert stats.mean_ci([2.0], seed=1) == (2.0, 2.0, "none")


def test_mean_ci_switches_method_by_size():
    small = stats.mean_ci([0.1 * i for i in range(20)], seed=1, b=50)
    assert small[2] == "bootstrap" and small[0] <= 0.95 <= small[1]
    big = stats.mean_ci([float(i % 7) for i in range(6000)], seed=1)
    assert big[2] == "normal" and big[0] < 3.0 < big[1]


@pytest.mark.parametrize("kw", [{"b": 0}, {"alpha": 0}, {"alpha": 1}])
def test_bootstrap_argument_validation(kw):
    with pytest.raises(ValueError):
        stats.bootstrap_ci([1.0, 2.0], seed=1, **kw)


def test_bootstrap_ci_of_the_mean_has_about_nominal_coverage():
    """Statistical self-check of the interval itself: 95 % nominal, allow 88-99 % over 300 draws."""
    rng = random.Random(11)
    hits = 0
    for i in range(300):
        xs = [rng.gauss(0.5, 1.0) for _ in range(30)]
        lo, hi = stats.bootstrap_ci(xs, seed=i, b=200)
        hits += lo <= 0.5 <= hi
    assert 0.88 <= hits / 300 <= 0.99


def test_mde_is_two_point_eight_standard_errors():
    xs = [0.0, 1.0, 2.0, 3.0]
    assert stats.mde(xs) == pytest.approx(2.8 * stats.stdev(xs) / 2)  # sqrt(4) = 2


def test_mde_scales_with_noise_and_shrinks_with_n():
    rng = random.Random(3)
    noisy = [rng.gauss(0, 1.0) for _ in range(100)]
    quiet = [x / 4 for x in noisy]
    assert stats.mde(noisy) == pytest.approx(4 * stats.mde(quiet))
    assert stats.mde(noisy[:25]) > stats.mde(noisy)
    assert stats.mde([1.0]) is None and stats.mde([]) is None
    assert stats.mde([0.5, 0.5, 0.5]) == 0.0  # no variance, no noise floor


@pytest.mark.parametrize(
    "a,b,p",
    [(0, 0, 1.0), (5, 5, 1.0), (0, 5, 0.0625), (5, 0, 0.0625), (1, 9, 0.021484375), (3, 4, 1.0)],
)
def test_mcnemar_exact_known_values(a, b, p):
    assert stats.mcnemar_exact(a, b) == pytest.approx(p)


@given(a=st.integers(0, 60), b=st.integers(0, 60))
def test_property_mcnemar_is_symmetric_and_a_probability(a, b):
    p = stats.mcnemar_exact(a, b)
    assert p == stats.mcnemar_exact(b, a) and 0.0 <= p <= 1.0


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError):
        stats.mcnemar_exact(-1, 2)


def test_spearman_known_values_ties_and_undefined_cases():
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert stats.spearman([1, 2, 3, 4, 5], [1, 3, 2, 5, 4]) == pytest.approx(0.8)
    assert stats.spearman([1, 1, 2, 3], [1, 2, 3, 3]) == pytest.approx(
        4 / 4.5
    )  # ties share average ranks
    assert stats.spearman([1, 2], [1, 2]) is None  # too few
    assert stats.spearman([1, 1, 1], [1, 2, 3]) is None  # no variance
    with pytest.raises(ValueError):
        stats.spearman([1, 2, 3], [1, 2])


@settings(deadline=None)
@given(xs=st.lists(floats, min_size=3, max_size=30), ys=st.lists(floats, min_size=3, max_size=30))
def test_property_spearman_is_bounded_and_invariant_to_monotone_transforms(xs, ys):
    n = min(len(xs), len(ys))
    xs, ys = xs[:n], ys[:n]
    r = stats.spearman(xs, ys)
    if r is None:
        return
    assert -1.0 <= r <= 1.0
    scaled = [x * 8 for x in xs]  # exact, strictly monotone: ranks are unchanged
    assert stats.spearman(scaled, ys) == pytest.approx(r, abs=1e-9)


def test_cohen_kappa():
    perfect = [("PASS", "PASS"), ("FAIL", "FAIL"), ("PASS", "PASS"), ("FAIL", "FAIL")]
    assert stats.cohen_kappa(perfect) == pytest.approx(1.0)
    opposite = [("PASS", "FAIL"), ("FAIL", "PASS")] * 3
    assert stats.cohen_kappa(opposite) == pytest.approx(-1.0)
    chance = [("PASS", "PASS"), ("PASS", "FAIL"), ("FAIL", "PASS"), ("FAIL", "FAIL")]
    assert stats.cohen_kappa(chance) == pytest.approx(0.0)
    assert stats.cohen_kappa([]) is None
    assert stats.cohen_kappa([("PASS", "PASS")] * 5) is None  # one label only: undefined
    # worked example: raters say P 8/7 times and F 2/3 times; observed 0.7, expected 0.62
    ex = [("P", "P")] * 6 + [("F", "F")] + [("P", "F")] * 2 + [("F", "P")]
    assert stats.cohen_kappa(ex) == pytest.approx((0.7 - 0.62) / (1 - 0.62))


def test_is_binary():
    assert stats.is_binary([0.0, 1.0, 1]) and not stats.is_binary([0.5]) and not stats.is_binary([])
