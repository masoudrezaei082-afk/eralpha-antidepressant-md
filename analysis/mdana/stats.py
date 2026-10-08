"""Statistics with the replica as the unit of replication (n = 3 independent 100-ns simulations per system).

Frames within a replica are correlated, so they are first averaged per replica; uncertainties are then the SEM of
the three replica means with a t-based 95 % CI (t(0.975, 2) = 4.303). Differences from estradiol use Welch's
unequal-variance interval. Block averaging (Flyvbjerg & Petersen 1989) is used only as a within-replica convergence
check, never to inflate n.
"""
import numpy as np
import pandas as pd
from scipy import stats as st


def summarize(x):
    """mean, SD, SEM and t-based 95 % CI of replica-level values."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, sem=np.nan, lo=np.nan, hi=np.nan)
    mean = x.mean()
    if n == 1:
        return dict(n=1, mean=mean, sd=np.nan, sem=np.nan, lo=np.nan, hi=np.nan)
    sd = x.std(ddof=1)
    sem = sd / np.sqrt(n)
    h = st.t.ppf(0.975, n - 1) * sem
    return dict(n=n, mean=mean, sd=sd, sem=sem, lo=mean - h, hi=mean + h)


def welch(a, b):
    """Difference of means a − b with Welch's 95 % CI and two-sided p (descriptive with n = 3)."""
    a, b = (np.asarray(v, dtype=float) for v in (a, b))
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else np.inf
    d = a.mean() - b.mean()
    h = st.t.ppf(0.975, df) * se
    p = 2 * st.t.sf(abs(d) / se, df) if se > 0 else np.nan
    return dict(diff=d, se=se, df=df, lo=d - h, hi=d + h, p=p)


def block_sem(x, min_blocks=4):
    """Flyvbjerg–Petersen blocking: SEM estimate at each blocking level and the plateau value.

    The plateau is taken as the maximum SEM over levels with at least `min_blocks` blocks; `converged` is True when
    the last two such levels agree within their own uncertainty (SEM/sqrt(2(n−1)))."""
    x = np.asarray(x, dtype=float)
    levels = []
    y = x.copy()
    while len(y) >= min_blocks:
        n = len(y)
        s = np.sqrt(y.var(ddof=1) / n)
        levels.append(dict(block=len(x) // n, n_blocks=n, sem=s, err=s / np.sqrt(2 * (n - 1))))
        y = 0.5 * (y[: n // 2 * 2: 2] + y[1: n // 2 * 2: 2])
    df = pd.DataFrame(levels)
    plateau = df["sem"].max()
    converged = bool(len(df) >= 2 and abs(df["sem"].iloc[-1] - df["sem"].iloc[-2]) <= df["err"].iloc[-1] + df["err"].iloc[-2])
    return df, plateau, converged


def cumulative_mean(x):
    x = np.asarray(x, dtype=float)
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def spearman(a, b):
    r, p = st.spearmanr(a, b)
    return float(r), float(p)


def ordering_supported(diffs):
    """An ordering against the reference is stated only if every interval (one per ε_in / method) excludes 0 on the
    same side. Returns 'lower', 'higher' or 'not resolved'."""
    los = np.array([d["lo"] for d in diffs])
    his = np.array([d["hi"] for d in diffs])
    if (his < 0).all():
        return "lower"
    if (los > 0).all():
        return "higher"
    return "not resolved"


def _selftest():
    """Hand-checked toy values: x = 1, 2, 3 -> mean 2, SD 1, SEM 1/sqrt(3) = 0.5774, CI half-width 4.3027 × 0.5774."""
    s = summarize([1, 2, 3])
    assert abs(s["mean"] - 2) < 1e-12 and abs(s["sd"] - 1) < 1e-12
    assert abs(s["sem"] - 0.5773503) < 1e-6
    assert abs(s["hi"] - (2 + 4.3026527 * 0.5773503)) < 1e-5
    w = welch([1, 2, 3], [1, 2, 3])
    assert abs(w["diff"]) < 1e-12 and w["lo"] < 0 < w["hi"]
    w = welch([4, 5, 6], [1, 2, 3])                     # equal variances: df = 4, se = sqrt(2/3)
    assert abs(w["df"] - 4) < 1e-9 and abs(w["se"] - np.sqrt(2 / 3)) < 1e-12
    assert abs(w["hi"] - (3 + 2.7764451 * np.sqrt(2 / 3))) < 1e-5
    rng = np.random.default_rng(0)
    white = rng.normal(size=4096)
    _, plat, _ = block_sem(white)
    assert abs(plat - 1 / 64) < 0.006, plat           # white noise: SEM ≈ 1/sqrt(4096)
    assert ordering_supported([dict(lo=-3, hi=-1), dict(lo=-2, hi=-0.1)]) == "lower"
    assert ordering_supported([dict(lo=-3, hi=-1), dict(lo=-2, hi=0.5)]) == "not resolved"
    return "stats self-test passed"


if __name__ == "__main__":
    print(_selftest())
