"""Statistics (docs/TARGET-ARCHITECTURE.md §9.1-9.3): stdlib only, pure, seeded, finite in and out.

Every function refuses NaN/Infinity input (a non-finite value here would silently poison an
aggregate; it is a bug upstream) and every resampling is seeded, so a report is reproducible.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence

Z95 = 1.959963984540054  # two-sided 95 %
MDE_FACTOR = 2.8  # ~ (z_{alpha/2} + z_power) at alpha 0.05, power 0.80
BOOTSTRAP_B = 2000
BOOTSTRAP_BELOW = 5000  # above this many values the normal approximation is used


def _finite(xs: Sequence[float], what: str = "values") -> None:
    for x in xs:
        if isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x):
            raise ValueError(f"{what} must be finite numbers, got {x!r}")


def mean(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("mean of no values")
    _finite(xs)
    return math.fsum(xs) / len(xs)


def stdev(xs: Sequence[float]) -> float:
    """Sample standard deviation (n - 1); 0.0 for a single value."""
    n = len(xs)
    if n == 0:
        raise ValueError("stdev of no values")
    if n == 1:
        _finite(xs)
        return 0.0
    m = mean(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (n - 1))


def quantile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (the common "type 7" definition)."""
    if not xs:
        raise ValueError("quantile of no values")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be within [0, 1]")
    _finite(xs)
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def median(xs: Sequence[float]) -> float:
    return quantile(xs, 0.5)


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion: well behaved at 0, 1 and small n."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n and n >= 1")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # rounding can leave the bound a hair inside the estimate (e.g. 0.9999999999999999 at k = n)
    return max(0.0, min(p, centre - half)), min(1.0, max(p, centre + half))


def normal_ci(xs: Sequence[float], z: float = Z95) -> tuple[float, float]:
    m = mean(xs)
    if len(xs) < 2:
        return m, m
    half = z * stdev(xs) / math.sqrt(len(xs))
    return m - half, m + half


def bootstrap_ci(
    xs: Sequence[float],
    *,
    seed: int,
    b: int = BOOTSTRAP_B,
    alpha: float = 0.05,
    stat: Callable[[Sequence[float]], float] = mean,
) -> tuple[float, float]:
    """Percentile bootstrap interval of `stat` (default the mean), seeded."""
    if not xs:
        raise ValueError("bootstrap of no values")
    if b < 1 or not 0 < alpha < 1:
        raise ValueError("b must be >= 1 and alpha in (0, 1)")
    _finite(xs)
    values = list(xs)
    n, rng = len(values), random.Random(seed)
    stats = sorted(stat(rng.choices(values, k=n)) for _ in range(b))
    lo = stats[min(b - 1, max(0, math.floor(alpha / 2 * b)))]
    hi = stats[min(b - 1, max(0, math.ceil((1 - alpha / 2) * b) - 1))]
    return lo, hi


def mean_ci(
    xs: Sequence[float],
    *,
    seed: int,
    b: int = BOOTSTRAP_B,
    bootstrap_below: int = BOOTSTRAP_BELOW,
) -> tuple[float, float, str]:
    """95 % interval for a mean and the method used: bootstrap for n < `bootstrap_below`, the
    normal approximation above it (where the CLT holds and the bootstrap would only cost time)."""
    if len(xs) < 2:
        m = mean(xs)
        return m, m, "none"
    if len(xs) < bootstrap_below:
        return (*bootstrap_ci(xs, seed=seed, b=b), "bootstrap")
    return (*normal_ci(xs), "normal")


def mde(diffs: Sequence[float]) -> float | None:
    """Minimum detectable difference, ~ 2.8 x SE of the paired differences (80 % power, 5 % alpha),
    from the observed variance. None when there are fewer than 2 pairs."""
    if len(diffs) < 2:
        return None
    return MDE_FACTOR * stdev(diffs) / math.sqrt(len(diffs))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts (binomial, p = 0.5)."""
    if only_a < 0 or only_b < 0:
        raise ValueError("counts must be non-negative")
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1  # average rank for ties
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation with average ranks for ties; None if undefined (n < 3 or no variance)."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 3:
        return None
    _finite(xs)
    _finite(ys)
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = math.fsum(rx) / len(rx), math.fsum(ry) / len(ry)
    sxx = math.fsum((a - mx) ** 2 for a in rx)
    syy = math.fsum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    return max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))


def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Cohen's kappa for two raters' labels; None when undefined (no pairs, or chance agreement
    is already 1 because both raters only ever used one label)."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(ca[label] * cb[label] for label in ca.keys() | cb.keys()) / (n * n)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def is_binary(xs: Sequence[float]) -> bool:
    """True when every value is exactly 0 or 1 (a proportion metric)."""
    return bool(xs) and all(x == 0 or x == 1 for x in xs)
