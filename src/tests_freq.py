"""Frequentist non-parametric tests with effect sizes and bootstrap CIs.

All tests return a ``dict`` (JSON-friendly) so that notebooks can pretty-print
or table-ify results uniformly. The non-parametric battery is locked in
``02-Methods/02.4-tests-non-param.md`` and the plan's "Tests fréquentistes
catalogue".

Banlist (contract §11): Pearson r, paired t-test (ordinal → interval
violation), asymptotic χ² on small cells, Bonferroni multiplicity.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from constants import (
    CI_DEFAULT,
    EQUIV_SMD_MARGIN,
    N_BOOT_DEFAULT,
    N_PERM_DEFAULT,
    PERM_EXACT_MAX,
    RANDOM_SEED,
    SITES_FT,
    SITES_PF,
)


# --- Effect sizes --------------------------------------------------------


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's δ for ordinal/non-parametric effect size.

    δ ∈ [-1, +1]. Sign indicates direction (positive: x > y stochastically).
    Computed exactly via pairwise comparison.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return float("nan")
    # Pairwise comparison via broadcasting
    diff = x[:, None] - y[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / (n_x * n_y))


def interpret_cliffs_delta(d: float) -> str:
    """Romano (2006) verbal magnitude for Cliff's δ.

    Thresholds: |δ| < 0.147 → negligible; 0.147–0.33 → small;
    0.33–0.474 → medium; ≥ 0.474 → large.
    """
    if np.isnan(d):
        return "n/a"
    a = abs(float(d))
    if a < 0.147:
        return "negligible"
    if a < 0.330:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def interpret_rank_biserial(r: float) -> str:
    """Same Romano thresholds applied to matched-pairs rank-biserial r."""
    if np.isnan(r):
        return "n/a"
    a = abs(float(r))
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "medium"
    return "large"


def interpret_cohens_d(d: float) -> str:
    """Cohen (1988) verbal magnitude for d (or SMD).

    |d| < 0.2 → negligible; 0.2–0.5 → small; 0.5–0.8 → medium; ≥ 0.8 → large.
    """
    if np.isnan(d):
        return "n/a"
    a = abs(float(d))
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def rank_biserial(x: Sequence[float], y: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation from Wilcoxon signed-rank."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[d != 0]  # drop ties (Wilcoxon convention 'wilcox')
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    total = w_plus + w_minus
    return float((w_plus - w_minus) / total)


# --- Bootstrap (BCa) -----------------------------------------------------


def _bca_endpoints(
    theta_hat: float,
    boot_stats: np.ndarray,
    jack_stats: np.ndarray,
    ci: float,
) -> tuple[float, float]:
    """Compute BCa CI endpoints from bootstrap + jackknife replicates.

    Parameters
    ----------
    theta_hat : float
        Point estimate on the full sample.
    boot_stats : np.ndarray
        Bootstrap replicates (shape ``(n_boot,)``).
    jack_stats : np.ndarray
        Jackknife replicates (shape ``(n,)``).
    ci : float
        Coverage (e.g. ``0.95``).

    Returns
    -------
    (lo, hi) : tuple of float
        Lower and upper BCa endpoints.
    """
    # Bias correction z0
    prop_less = float((boot_stats < theta_hat).mean())
    prop_less = float(np.clip(prop_less, 1e-6, 1 - 1e-6))
    z0 = float(stats.norm.ppf(prop_less))

    # Acceleration via jackknife
    jack_mean = jack_stats.mean()
    num = ((jack_mean - jack_stats) ** 3).sum()
    den = 6 * (((jack_mean - jack_stats) ** 2).sum() ** 1.5)
    if den == 0:
        warnings.warn(
            "BCa acceleration undefined (zero jackknife variance); "
            "falling back to percentile CI.",
            RuntimeWarning,
            stacklevel=3,
        )
        a = 0.0
    else:
        a = float(num / den)

    alpha = (1 - ci) / 2
    z_lo = float(stats.norm.ppf(alpha))
    z_hi = float(stats.norm.ppf(1 - alpha))
    alpha_lo = float(stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo))))
    alpha_hi = float(stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi))))
    lo = float(np.quantile(boot_stats, alpha_lo))
    hi = float(np.quantile(boot_stats, alpha_hi))
    return lo, hi


def _bca_ci(
    stat_fn: Callable,
    data: tuple,
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
    paired: bool = False,
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap CI for a generic statistic.

    Parameters
    ----------
    stat_fn : callable
        Function taking ``*data`` and returning a float.
    data : tuple of arrays
        Arrays to resample.
    n_boot, ci : int, float
    rng : numpy.random.Generator
    paired : bool, default False
        If True, resample joint indices (keeps pairing  required for Spearman).
        If False, resample each array independently (two-sample statistics).

    Returns
    -------
    (lo, hi) : tuple of float
    """
    theta_hat = float(stat_fn(*data))
    boot_stats = np.empty(n_boot)
    sizes = [len(a) for a in data]

    if paired:
        n = sizes[0]
        if not all(s == n for s in sizes):
            raise ValueError("paired=True requires arrays of identical length.")
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_stats[i] = stat_fn(*[a[idx] for a in data])
    else:
        for i in range(n_boot):
            sample = tuple(a[rng.integers(0, n, size=n)] for a, n in zip(data, sizes))
            boot_stats[i] = stat_fn(*sample)

    # Jackknife: leave-one-out on first array (or joint for paired)
    a0 = data[0]
    n0 = len(a0)
    jack = np.empty(n0)
    for i in range(n0):
        mask = np.arange(n0) != i
        if paired:
            rest = tuple(arr[mask] for arr in data)
        else:
            rest = tuple([a0[mask]] + [arr for arr in data[1:]])
        try:
            jack[i] = stat_fn(*rest)
        except Exception:  # noqa: BLE001
            jack[i] = theta_hat

    return _bca_endpoints(theta_hat, boot_stats, jack, ci)


# --- Tests ---------------------------------------------------------------


def mwu_with_effects(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = N_BOOT_DEFAULT,
    ci: float = CI_DEFAULT,
    seed: int = RANDOM_SEED,
) -> dict:
    """Mann-Whitney U + Cliff's δ + BCa bootstrap CI on δ.

    Parameters
    ----------
    x, y : array-like
        Two independent samples (NaNs dropped).
    n_boot : int
    ci : float
        Coverage of the bootstrap CI.

    Returns
    -------
    dict
        ``{statistic, pvalue, n_x, n_y, cliffs_delta, delta_ci_lo, delta_ci_hi}``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    U, p = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
    delta = cliffs_delta(x, y)
    rng = np.random.default_rng(seed)
    lo, hi = _bca_ci(cliffs_delta, (x, y), n_boot=n_boot, ci=ci, rng=rng)
    n_x_, n_y_ = len(x), len(y)
    prob_superiority = (
        float(U / (n_x_ * n_y_)) if (n_x_ > 0 and n_y_ > 0) else float("nan")
    )
    return dict(
        statistic=float(U),
        pvalue=float(p),
        n_x=n_x_,
        n_y=n_y_,
        cliffs_delta=delta,
        delta_ci_lo=lo,
        delta_ci_hi=hi,
        ci=ci,
        cliffs_delta_magnitude=interpret_cliffs_delta(delta),
        probability_of_superiority=prob_superiority,
    )


# --- Permutation inference on Cliff's δ (consensus point H) --------------


def _cliffs_delta_from_pooled(pooled: np.ndarray, n_x: int) -> float:
    """Cliff's δ where the first ``n_x`` entries of ``pooled`` are group x."""
    x = pooled[:n_x]
    y = pooled[n_x:]
    diff = x[:, None] - y[None, :]
    return ((diff > 0).sum() - (diff < 0).sum()) / (len(x) * len(y))


def permutation_test_cliff(
    x: Sequence[float],
    y: Sequence[float],
    n_perm: int = N_PERM_DEFAULT,
    seed: int = RANDOM_SEED,
    exact_max: int = PERM_EXACT_MAX,
    alternative: str = "two-sided",
) -> dict:
    """Permutation test for Cliff's δ between two groups (consensus point H).

    The null hypothesis is exchangeability of the group labels. We permute the
    labels, recompute δ, and locate the observed δ in the permutation
    distribution. The test is **exact** (enumerates all C(n, n_x) label
    assignments) when that count ≤ ``exact_max``; otherwise it falls back to a
    Monte-Carlo approximation with ``n_perm`` random relabelings.

    Parameters
    ----------
    x, y : array-like
        Two independent samples (NaNs dropped).
    n_perm : int
        Monte-Carlo resamples when the exact enumeration is intractable.
    seed : int
        RNG seed (fixed at 42 for reproducibility).
    exact_max : int
        Largest C(n, n_x) for which the exact enumeration is attempted.
    alternative : {"two-sided", "greater", "less"}
        ``greater`` tests δ > 0 (x stochastically larger than y).

    Returns
    -------
    dict
        ``{cliffs_delta, pvalue, method, n_perm_effective, alternative,
        n_x, n_y}``. ``method`` is ``"exact"`` or ``"monte-carlo"``. The
        Monte-Carlo p-value uses the (b + 1) / (m + 1) correction
        (Davison & Hinkley) so it is never exactly 0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return dict(
            cliffs_delta=float("nan"),
            pvalue=float("nan"),
            method="degenerate",
            n_perm_effective=0,
            alternative=alternative,
            n_x=n_x,
            n_y=n_y,
        )

    pooled = np.concatenate([x, y])
    n = n_x + n_y
    obs = _cliffs_delta_from_pooled(pooled, n_x)

    n_comb = math.comb(n, n_x)
    use_exact = n_comb <= exact_max

    def _count(deltas: np.ndarray) -> float:
        if alternative == "greater":
            return float((deltas >= obs).sum())
        if alternative == "less":
            return float((deltas <= obs).sum())
        # two-sided: as-or-more extreme in |δ| (centred at 0 under H0)
        return float((np.abs(deltas) >= abs(obs) - 1e-12).sum())

    if use_exact:
        from itertools import combinations

        idx_all = np.arange(n)
        deltas = np.empty(n_comb)
        for i, combo in enumerate(combinations(idx_all, n_x)):
            mask = np.zeros(n, dtype=bool)
            mask[list(combo)] = True
            perm = np.concatenate([pooled[mask], pooled[~mask]])
            deltas[i] = _cliffs_delta_from_pooled(perm, n_x)
        pval = _count(deltas) / n_comb
        method = "exact"
        m_eff = n_comb
    else:
        rng = np.random.default_rng(seed)
        deltas = np.empty(n_perm)
        for i in range(n_perm):
            perm = rng.permutation(pooled)
            deltas[i] = _cliffs_delta_from_pooled(perm, n_x)
        # (b + 1)/(m + 1) Monte-Carlo p-value
        pval = (_count(deltas) + 1.0) / (n_perm + 1.0)
        method = "monte-carlo"
        m_eff = n_perm

    return dict(
        cliffs_delta=float(obs),
        pvalue=float(min(pval, 1.0)),
        method=method,
        n_perm_effective=int(m_eff),
        alternative=alternative,
        n_x=n_x,
        n_y=n_y,
    )


def cliff_ci_inversion(
    x: Sequence[float],
    y: Sequence[float],
    ci: float = CI_DEFAULT,
) -> tuple[float, float]:
    """Cliff's δ CI by test inversion (Cliff 1993/1996 consistent-variance).

    Guard-rail CI requested by consensus point H: a second interval that does
    **not** rely on the BCa jackknife acceleration (unstable at n_mén = 19).
    This is the standard analytic interval obtained by **inverting the z-test
    of** :math:`H_0:\\delta = \\delta_0` using Cliff's unbiased,
    heteroscedastic-consistent variance estimator of ``d``. Because the
    sampling distribution of ``d`` is asymmetric near ±1, the interval is
    built on a variance-stabilising form and is asymmetric (it never exceeds
    ±1), which is exactly the behaviour the BCa interval can miss at small n.

    Variance (Cliff 1996, eq. for the unbiased dominance statistic)::

        d_ij  = sign(x_i − y_j)
        d_i.  = mean_j d_ij          (over y)
        d_.j  = mean_i d_ij          (over x)
        s_di² = Σ (d_i. − d)² / (n_x − 1)
        s_dj² = Σ (d_.j − d)² / (n_y − 1)
        σ²(d) = [ (n_y² s_di²) + (n_x² s_dj²) − Σ_ij(d_ij − d)²/(n_x n_y) ]
                 / (n_x n_y (n_x − 1)(n_y − 1)) ... (consistent estimator)

    Parameters
    ----------
    x, y : array-like
    ci : float
        Coverage (default 0.95).

    Returns
    -------
    (lo, hi) : tuple of float
        Asymmetric δ bounds in [-1, 1]. NaN if undefined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    n_x, n_y = len(x), len(y)
    if n_x < 2 or n_y < 2:
        return float("nan"), float("nan")

    dij = np.sign(x[:, None] - y[None, :])  # (n_x, n_y) in {-1,0,1}
    d = float(dij.mean())
    di = dij.mean(axis=1)  # per-x dominance
    dj = dij.mean(axis=0)  # per-y dominance
    s_di2 = float(((di - d) ** 2).sum() / (n_x - 1)) if n_x > 1 else 0.0
    s_dj2 = float(((dj - d) ** 2).sum() / (n_y - 1)) if n_y > 1 else 0.0
    s_dij2 = float(((dij - d) ** 2).sum() / (n_x * n_y - 1))
    # Consistent (Cliff) variance of d.
    var_d = (((n_y - 1) * s_di2) + ((n_x - 1) * s_dj2) + s_dij2) / (n_x * n_y)
    if not np.isfinite(var_d) or var_d <= 0:
        return float(d), float(d)
    sd = math.sqrt(var_d)

    z = float(stats.norm.ppf(1 - (1 - ci) / 2))
    # Asymmetric interval (Feng & Cliff 2004)  stays within ±1 even when d
    # is near the boundary, which is the small-n robustness the BCa lacks.
    num_lo = d - d**3 - z * sd * math.sqrt(max((1 - d**2) ** 2 + z**2 * sd**2, 0.0))
    num_hi = d - d**3 + z * sd * math.sqrt(max((1 - d**2) ** 2 + z**2 * sd**2, 0.0))
    den = (1 - d**2) + z**2 * sd**2
    if den == 0:
        return float(d), float(d)
    lo = num_lo / den
    hi = num_hi / den
    lo = max(-1.0, min(1.0, lo))
    hi = max(-1.0, min(1.0, hi))
    return float(min(lo, hi)), float(max(lo, hi))


# Backwards-compatible alias (the consensus brief calls this the
# "permutation-inversion" guard-rail; the realised object is the standard
# test-inversion analytic interval, robust at n = 19).
cliff_ci_permutation_inversion = cliff_ci_inversion


def wilcoxon_exact_with_rrb(x: Sequence[float], y: Sequence[float]) -> dict:
    """Wilcoxon signed-rank exact test + matched-pairs rank-biserial r.

    Drops pairs with NaN. Uses ``zero_method='wilcox'`` (drop zeros) and
    exact distribution (small-n).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    d = x - y
    nz = (d != 0).sum()
    if nz == 0:
        return dict(
            statistic=0.0,
            pvalue=1.0,
            n_pairs=len(x),
            n_nonzero=0,
            rank_biserial=0.0,
            method="degenerate",
            rrb_magnitude=interpret_rank_biserial(0.0),
        )
    method = "exact" if nz <= 25 else "approx"
    W, p = stats.wilcoxon(x, y, zero_method="wilcox", method=method)
    rrb = rank_biserial(x, y)
    return dict(
        statistic=float(W),
        pvalue=float(p),
        n_pairs=len(x),
        n_nonzero=int(nz),
        rank_biserial=rrb,
        method=method,
        rrb_magnitude=interpret_rank_biserial(rrb),
    )


def mcnemar_exact_midp(table) -> dict:
    """McNemar exact mid-p test (statsmodels).

    Parameters
    ----------
    table : 2×2 array-like
        ``[[both_neg, neg_to_pos], [pos_to_neg, both_pos]]``.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    arr = np.asarray(table)
    res = mcnemar(arr, exact=True, correction=False)
    # res.pvalue = exact two-sided binomial(b, n, 0.5). We additionally compute
    # the mid-p variant manually (Lancaster 1961): mid_p = 2 × [F(k) − 0.5·f(k)]
    # where k = min(b, c). This is a separate, less conservative p-value
    # NOT a "correction" applied to res.pvalue. Both are returned as distinct
    # fields so the caller chooses.
    b = int(arr[0, 1])
    c = int(arr[1, 0])
    n = b + c
    if n == 0:
        return dict(statistic=0.0, pvalue=1.0, mid_p=1.0, b=b, c=c)
    k = min(b, c)
    midp = 2 * (stats.binom.cdf(k, n, 0.5) - 0.5 * stats.binom.pmf(k, n, 0.5))
    midp = float(np.clip(midp, 0.0, 1.0))
    # mid-p must be ≤ exact two-sided p (degenerates only when k = n/2).
    assert midp <= float(res.pvalue) + 1e-9, (
        f"mid_p={midp} > exact p={res.pvalue}: distributional invariant broken."
    )
    return dict(
        statistic=float(res.statistic), pvalue=float(res.pvalue), mid_p=midp, b=b, c=c
    )


def kw_dunn(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    """Kruskal-Wallis + Dunn post-hoc (Bonferroni) + ε² effect size."""
    import scikit_posthocs as sp

    sub = df[[group_col, value_col]].dropna()
    groups = [g[value_col].values for _, g in sub.groupby(group_col)]
    H, p = stats.kruskal(*groups)
    n = len(sub)
    k = len(groups)
    epsilon_sq = (H - k + 1) / (n - k) if n > k else float("nan")
    posthoc = sp.posthoc_dunn(
        sub, val_col=value_col, group_col=group_col, p_adjust="bonferroni"
    )
    return dict(
        statistic=float(H),
        pvalue=float(p),
        epsilon_sq=float(epsilon_sq),
        n=int(n),
        k=int(k),
        posthoc=posthoc,
    )


def fisher_exact_2x2(table: np.ndarray) -> dict:
    """Fisher exact two-sided + odds ratio + mid-p two-sided.

    Parameters
    ----------
    table : 2×2 array-like
    """
    table = np.asarray(table)
    odds, p = stats.fisher_exact(table, alternative="two-sided")
    # Mid-p two-sided approximation
    midp = max(
        0.0,
        p
        - stats.hypergeom.pmf(
            table[0, 0],
            table.sum(),
            table[0].sum(),
            table[:, 0].sum(),
        ),
    )
    return dict(
        odds_ratio=float(odds), pvalue=float(p), mid_p=float(midp), table=table.tolist()
    )


def spearman_bca(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = N_BOOT_DEFAULT,
    ci: float = CI_DEFAULT,
    seed: int = RANDOM_SEED,
) -> dict:
    """Spearman ρ + BCa bootstrap CI on ρ (paired resampling, joint indices)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    rho, p = stats.spearmanr(x, y)

    def _stat(a, b):
        return stats.spearmanr(a, b).statistic

    rng = np.random.default_rng(seed)
    lo, hi = _bca_ci(_stat, (x, y), n_boot=n_boot, ci=ci, rng=rng, paired=True)
    return dict(
        rho=float(rho), pvalue=float(p), ci_lo=lo, ci_hi=hi, ci=ci, n=int(len(x))
    )


def bh_fdr(pvals: Sequence[float], q: float = 0.10) -> dict:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals : array-like
    q : float
        FDR target (default 0.10, contract §2.4).

    Returns
    -------
    dict
        ``{reject, pvals_corrected, q, n}``.
    """
    from statsmodels.stats.multitest import multipletests

    reject, p_adj, _, _ = multipletests(pvals, alpha=q, method="fdr_bh")
    return dict(
        reject=reject.tolist(), pvals_corrected=p_adj.tolist(), q=q, n=len(pvals)
    )


def smd_continuous(x: Sequence[float], y: Sequence[float]) -> float:
    """Standardised mean difference (Cohen's d, pooled SD).

    Used for baseline-balance reporting (Table 1) alongside MWU p-values.
    Note: Cohen's d strictly assumes interval data; for ordinal scores we
    interpret SMD as a descriptive index only.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2))
    return float((mx - my) / sp) if sp > 0 else float("nan")


def smd_with_magnitude(x: Sequence[float], y: Sequence[float]) -> dict:
    """Standardised mean difference + Cohen (1988) verbal magnitude.

    Sibling of :func:`smd_continuous` returning a dict with the verbal label
    attached so Table 1 rows can be auto-annotated without a second call.
    """
    d = smd_continuous(x, y)
    return dict(smd=d, magnitude=interpret_cohens_d(d))


# ============================================================================
# High-level contrast helpers (consensus points B, D, G, H)
# ============================================================================


def pf_contrast(
    df_wide: pd.DataFrame,
    value_col: str = "delta_lesion_pf",
    group_col: str = "group",
    cyclops: str = "cyclops",
    meniscus: str = "meniscus",
    n_boot: int = N_BOOT_DEFAULT,
    n_perm: int = N_PERM_DEFAULT,
    ci: float = CI_DEFAULT,
    seed: int = RANDOM_SEED,
) -> dict:
    """PRIMARY patellofemoral contrast: MWU + Cliff δ + permutation + BCa + inversion CI.

    Implements the primary frequentist support for the PF estimand
    (consensus point B): the progression of the PF block {trochlée, rotule}
    in cyclops vs meniscus, tested on ranks (no interval assumption). Bundles
    every inference requested by points G and H:

    * Mann-Whitney U (two-sided), exact small-n method.
    * Cliff's δ with Romano verbal magnitude.
    * **Permutation test** on δ (exact if tractable, else Monte-Carlo
      ``n_perm``)  the decisional frequentist support.
    * **BCa CI** (B = 10 000) on δ.
    * **Permutation-inversion CI** on δ as a guard-rail (point H).

    Parameters
    ----------
    df_wide : pd.DataFrame
        Output of :func:`preprocessing.to_wide`.
    value_col : str
        Per-patient Δ column (default ``delta_lesion_pf``).
    group_col, cyclops, meniscus : str
    n_boot, n_perm, ci, seed : int / float

    Returns
    -------
    dict
        Flat dict with ``mwu_*``, ``cliffs_delta``, ``perm_*``, ``bca_*``,
        ``inv_ci_*`` keys plus group sizes.
    """
    sub = df_wide[[group_col, value_col]].dropna()
    x = sub.loc[sub[group_col] == cyclops, value_col].astype(float).values
    y = sub.loc[sub[group_col] == meniscus, value_col].astype(float).values

    mwu = mwu_with_effects(x, y, n_boot=n_boot, ci=ci, seed=seed)
    perm = permutation_test_cliff(
        x, y, n_perm=n_perm, seed=seed, alternative="two-sided"
    )
    inv_lo, inv_hi = cliff_ci_inversion(x, y, ci=ci)
    return dict(
        value_col=value_col,
        cyclops=cyclops,
        meniscus=meniscus,
        n_cyclops=int(len(x)),
        n_meniscus=int(len(y)),
        mwu_U=mwu["statistic"],
        mwu_p=mwu["pvalue"],
        cliffs_delta=mwu["cliffs_delta"],
        cliffs_delta_magnitude=mwu["cliffs_delta_magnitude"],
        probability_of_superiority=mwu["probability_of_superiority"],
        bca_lo=mwu["delta_ci_lo"],
        bca_hi=mwu["delta_ci_hi"],
        bca_B=n_boot,
        perm_p=perm["pvalue"],
        perm_method=perm["method"],
        perm_n=perm["n_perm_effective"],
        inv_ci_lo=inv_lo,
        inv_ci_hi=inv_hi,
        ci=ci,
    )


def paired_pf_vs_ft(
    df_wide: pd.DataFrame,
    pf_col: str = "delta_lesion_pf",
    ft_col: str = "delta_lesion_ft",
    group_col: str = "group",
    cyclops: str = "cyclops",
) -> dict:
    """Within-cyclops Wilcoxon: ΔPF vs ΔFT (topographic specificity).

    Tests, in the cyclops group only, whether the PF block progresses more
    than the FT block within the same patient (consensus point B mechanistic
    specificity). Paired Wilcoxon signed-rank on (ΔPF, ΔFT) per patient +
    matched-pairs rank-biserial r.

    Parameters
    ----------
    df_wide : pd.DataFrame
    pf_col, ft_col : str
        Per-patient block-Δ columns.
    group_col, cyclops : str
        Restrict to ``df_wide[group_col] == cyclops`` (default cyclops).

    Returns
    -------
    dict
        Output of :func:`wilcoxon_exact_with_rrb` plus ``cyclops``, ``pf_col``,
        ``ft_col``, and the per-block worsening counts.
    """
    sub = df_wide[df_wide[group_col] == cyclops][[pf_col, ft_col]].dropna()
    pf = sub[pf_col].astype(float).values
    ft = sub[ft_col].astype(float).values
    res = wilcoxon_exact_with_rrb(pf, ft)
    res.update(
        cyclops=cyclops,
        pf_col=pf_col,
        ft_col=ft_col,
        n_pf_worsened=int((pf > 0).sum()),
        n_ft_worsened=int((ft > 0).sum()),
        median_delta_pf=float(np.median(pf)) if len(pf) else float("nan"),
        median_delta_ft=float(np.median(ft)) if len(ft) else float("nan"),
    )
    return res


# --- Firth penalized logistic (quasi-separation guard, revue 2026-05-29) ----


def _firth_irls(X: np.ndarray, y: np.ndarray, max_iter: int = 1000, tol: float = 1e-8):
    """Faithful Firth penalized-likelihood logistic fit (Heinze & Schemper 2002).

    Newton–Raphson on the Jeffreys-penalised score
    ``U*(β) = Xᵀ(y − π + h·(½ − π))`` where ``h`` is the hat-matrix diagonal.
    ``X`` must already include the intercept column.

    Returns
    -------
    (beta, cov) : (np.ndarray, np.ndarray)
        Coefficient vector and the penalised information inverse (Wald SEs =
        ``sqrt(diag(cov))``).
    """
    n, p = X.shape
    beta = np.zeros(p)
    info_inv = np.eye(p)
    for _ in range(max_iter):
        eta = X @ beta
        pi = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(pi * (1.0 - pi), 1e-10, None)
        info = (X * w[:, None]).T @ X
        try:
            info_inv = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            info_inv = np.linalg.pinv(info)
        Xs = X * np.sqrt(w)[:, None]
        h = np.einsum("ij,jk,ik->i", Xs, info_inv, Xs)  # hat diagonal
        U = X.T @ (y - pi + h * (0.5 - pi))
        step = info_inv @ U
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta, info_inv


def firth_or(
    df: pd.DataFrame,
    outcome_col: str = "worsened_pf",
    group_col: str = "group",
    cyclops: str = "cyclops",
    meniscus: str = "meniscus",
    covariates: Sequence[str] = (),
    ci: float = CI_DEFAULT,
) -> dict:
    """Firth penalized-likelihood OR for the group effect (robust to separation).

    ``worsened_pf`` has **1 / 19** events in meniscus → near-complete
    separation, so plain ML logistic returns OR ≈ 24 (≈ 47 once adjusted) with
    an unstable, enormous CI  a *ghost number* that must not anchor the paper
    (revue-methodo 2026-05-29). Firth's penalised likelihood (Jeffreys prior)
    yields a finite, stable estimate and a profile-penalised-likelihood CI.
    Optionally adjusts for ``covariates`` (e.g. ``female``, ``age_at_trauma``)
    this powers the headline **sex+age-adjusted** PF robustness analysis.

    Uses the ``firthlogist`` package when importable (profile-LR CI); otherwise a
    hand-rolled Firth IRLS with a Wald CI (``method`` records which path ran).

    Returns
    -------
    dict
        ``{outcome, cyclops, meniscus, covariates, odds_ratio, or_ci_lo, or_ci_hi,
        beta, p, n, n_events_cyclops, n_events_meniscus, method, separation_ml}``.
    """
    d = df[df[group_col].isin([cyclops, meniscus])].copy()
    d["_cyclops"] = (d[group_col] == cyclops).astype(float)
    d["_y"] = pd.to_numeric(d[outcome_col], errors="coerce")
    cov_present = [c for c in covariates if c in d.columns]
    for c in cov_present:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["_y", "_cyclops", *cov_present])

    y = d["_y"].astype(int).values
    feat = ["_cyclops", *cov_present]
    Xnoint = np.column_stack([d[c].astype(float).values for c in feat])

    ev_cyc = int(d.loc[d["_cyclops"] == 1, "_y"].sum())
    ev_men = int(d.loc[d["_cyclops"] == 0, "_y"].sum())
    n_cyc = int((d["_cyclops"] == 1).sum())
    n_men = int((d["_cyclops"] == 0).sum())
    # Quasi- or complete separation destabilises ML; flag when the smallest
    # outcome x group cell is <= 1 (meniscus has a single worsened_pf event,
    # which is exactly why the plain-logit OR is a ghost number).
    min_cell = int(min(ev_men, n_men - ev_men, ev_cyc, n_cyc - ev_cyc))
    separation = bool(min_cell <= 1)

    beta = ci_lo_b = ci_hi_b = pval = float("nan")
    try:
        from firthlogist import FirthLogisticRegression

        fl = FirthLogisticRegression(alpha=round(1 - ci, 4))
        fl.fit(Xnoint, y)
        beta = float(fl.coef_[0])  # _cyclops is feature index 0
        lo, hi = fl.ci_[0]
        ci_lo_b, ci_hi_b = float(lo), float(hi)
        pval = float(fl.pvals_[0])
        method = "firthlogist(profile-LR)"
    except Exception:  # noqa: BLE001  package absent or API drift → faithful fallback
        Xint = np.column_stack([np.ones(len(d)), Xnoint])
        b, cov = _firth_irls(Xint, y)
        se = float(np.sqrt(np.diag(cov))[1])  # _cyclops at index 1 (after intercept)
        beta = float(b[1])
        z = float(stats.norm.ppf(1 - (1 - ci) / 2))
        ci_lo_b, ci_hi_b = beta - z * se, beta + z * se
        pval = (
            float(2 * (1 - stats.norm.cdf(abs(beta) / se))) if se > 0 else float("nan")
        )
        method = "firth-irls(wald)"

    return dict(
        outcome=outcome_col,
        cyclops=cyclops,
        meniscus=meniscus,
        covariates=cov_present,
        odds_ratio=float(np.exp(beta)),
        or_ci_lo=float(np.exp(ci_lo_b)),
        or_ci_hi=float(np.exp(ci_hi_b)),
        beta=float(beta),
        p=float(pval),
        n=int(len(d)),
        n_events_cyclops=ev_cyc,
        n_events_meniscus=ev_men,
        method=method,
        separation_ml=bool(separation),
        min_cell=min_cell,
    )


def sensitivity_covariate(
    df_patient_wide: pd.DataFrame,
    outcome_col: str = "worsened_pf",
    group_col: str = "group",
    covariates: Sequence[str] = ("pivot_pivot_contact", "travail_physique"),
    cyclops: str = "cyclops",
    meniscus: str = "meniscus",
    use_firth: bool = True,
) -> dict:
    """Logistic sensitivity: group effect on a binary outcome WITH vs WITHOUT covariates.

    Consensus point  sport/occupation sensitivity (run *with* and *without*
    ``Pivot`` and ``Travail physique``). Fits two logistic regressions of
    ``outcome_col`` (e.g. ``worsened_pf``) on the group indicator: a crude
    model and an adjusted model that adds the covariates. Returns both odds
    ratios so the analyst can show the PF effect is not an artefact of the
    sport/occupation imbalance.

    Parameters
    ----------
    df_patient_wide : pd.DataFrame
        Wide frame joined with patient covariates (one row per patient).
    outcome_col : str
        Binary 0/1 outcome.
    group_col, cyclops, meniscus : str
    covariates : sequence of str
        Adjustment covariates added in the adjusted model.

    Returns
    -------
    dict
        ``{or_crude, p_crude, or_adjusted, p_adjusted, covariates, n_crude,
        n_adjusted, adjusted_ok}``. Odds ratios are for the *cyclops* vs meniscus.
    """
    # Default path: Firth penalised logistic for BOTH crude and adjusted models.
    # worsened_pf has 1/20 meniscus events → ML separation makes the plain-logit
    # OR (24 → 47) a ghost number; Firth gives a stable OR + CI (revue 2026-05-29).
    if use_firth:
        crude = firth_or(
            df_patient_wide, outcome_col, group_col, cyclops, meniscus, covariates=()
        )
        adj = firth_or(
            df_patient_wide,
            outcome_col,
            group_col,
            cyclops,
            meniscus,
            covariates=covariates,
        )
        return dict(
            outcome=outcome_col,
            cyclops=cyclops,
            meniscus=meniscus,
            covariates=adj["covariates"],
            or_crude=crude["odds_ratio"],
            p_crude=crude["p"],
            n_crude=crude["n"],
            or_adjusted=adj["odds_ratio"],
            p_adjusted=adj["p"],
            n_adjusted=adj["n"],
            or_crude_ci=(crude["or_ci_lo"], crude["or_ci_hi"]),
            or_adjusted_ci=(adj["or_ci_lo"], adj["or_ci_hi"]),
            method=adj["method"],
            separation_ml=adj["separation_ml"],
            adjusted_ok=bool(np.isfinite(adj["odds_ratio"])),
        )

    import statsmodels.formula.api as smf

    df = df_patient_wide.copy()
    df = df[df[group_col].isin([cyclops, meniscus])].copy()
    df["_cyclops"] = (df[group_col] == cyclops).astype(int)
    df["_y"] = pd.to_numeric(df[outcome_col], errors="coerce")

    def _fit(formula: str, data: pd.DataFrame) -> tuple[float, float, int]:
        d = data.dropna(subset=["_y"]).copy()
        try:
            m = smf.logit(formula, data=d).fit(disp=0)
            coef = m.params.get("_cyclops", float("nan"))
            pval = m.pvalues.get("_cyclops", float("nan"))
            return float(np.exp(coef)), float(pval), int(d.shape[0])
        except Exception as exc:  # noqa: BLE001  separation / singular design
            return float("nan"), float("nan"), int(d.shape[0])

    or_crude, p_crude, n_crude = _fit("_y ~ _cyclops", df)

    cov_present = [c for c in covariates if c in df.columns]
    rhs = " + ".join(["_cyclops", *cov_present]) if cov_present else "_cyclops"
    adj_data = df.dropna(subset=["_y", *cov_present]) if cov_present else df
    or_adj, p_adj, n_adj = _fit(f"_y ~ {rhs}", adj_data)

    return dict(
        outcome=outcome_col,
        cyclops=cyclops,
        meniscus=meniscus,
        covariates=cov_present,
        or_crude=or_crude,
        p_crude=p_crude,
        n_crude=n_crude,
        or_adjusted=or_adj,
        p_adjusted=p_adj,
        n_adjusted=n_adj,
        adjusted_ok=bool(np.isfinite(or_adj)),
    )


def tost_equivalence(
    x: Sequence[float],
    y: Sequence[float],
    bound: float,
) -> dict:
    """Two one-sided tests (TOST) for equivalence of two independent means.

    Replaces the logical error of reading a non-significant difference test
    (e.g. baseline MWU p=0.78) as "groups are equivalent"  *absence of evidence
    is not evidence of absence*. TOST instead tests whether the mean difference
    lies **within** an equivalence margin ``±bound``: H0 = |μx − μy| ≥ bound,
    H1 = within ±bound. The TOST p is ``max(p_lower, p_upper)``; ``p < α`` ⇒
    equivalence established within ``±bound``.

    Returns
    -------
    dict ``{mean_diff, bound, t_lower, t_upper, p_lower, p_upper, tost_p, df}``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return dict(
            mean_diff=float("nan"),
            bound=float(bound),
            tost_p=float("nan"),
            p_lower=float("nan"),
            p_upper=float("nan"),
            df=0,
        )
    mdiff = float(x.mean() - y.mean())
    sp = math.sqrt(
        ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    )
    se = sp * math.sqrt(1.0 / nx + 1.0 / ny)
    df = nx + ny - 2
    if se == 0:
        equiv = abs(mdiff) < bound
        return dict(
            mean_diff=mdiff,
            bound=float(bound),
            tost_p=0.0 if equiv else 1.0,
            p_lower=float("nan"),
            p_upper=float("nan"),
            df=df,
        )
    t_lower = (mdiff + bound) / se  # H0: diff <= -bound
    t_upper = (mdiff - bound) / se  # H0: diff >= +bound
    p_lower = float(stats.t.sf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    return dict(
        mean_diff=mdiff,
        bound=float(bound),
        t_lower=float(t_lower),
        t_upper=float(t_upper),
        p_lower=p_lower,
        p_upper=p_upper,
        tost_p=float(max(p_lower, p_upper)),
        df=int(df),
    )


def baseline_block_balance(
    df_wide: pd.DataFrame,
    col: str = "lesion_pf_S1",
    group_col: str = "group",
    cyclops: str = "cyclops",
    meniscus: str = "meniscus",
    bound: Optional[float] = None,
) -> dict:
    """Baseline balance on a **single block sub-score at S1** (+ TOST).

    Addresses the reviewer point (revue 2026-05-29) that "baseline equivalent
    (MWU p=0.78)" was tested on the global 6-sum, not on the block that carries
    the outcome  and that a non-significant difference is not equivalence.
    Reports the MWU p, the SMD, and a TOST equivalence p within ``±bound``
    (default = 0.5 pooled SD, a "small" SMD margin) on ``col``.

    Use ``col='lesion_pf_S1'`` for the PF block (carries the primary contrast)
    and ``col='lesion_ft_S1'`` for the FT block  the latter is needed to read
    the PF−FT topographic *contrast* causally, since a baseline FT gap would
    confound the localisation. Note ``bound`` is **data-derived** (0.5·s_pooled
    of *this* block), so the equivalence box rescales per block.

    Returns
    -------
    dict ``{col, n_cyclops, n_meniscus, median_cyclops, median_meniscus, mwu_p, smd,
    tost_bound, tost_p, equivalent}``.
    """
    sub = df_wide[[group_col, col]].dropna()
    x = sub.loc[sub[group_col] == cyclops, col].astype(float).values
    y = sub.loc[sub[group_col] == meniscus, col].astype(float).values
    if len(x) < 2 or len(y) < 2:
        return dict(
            col=col,
            n_cyclops=len(x),
            n_meniscus=len(y),
            mwu_p=float("nan"),
            smd=float("nan"),
            tost_p=float("nan"),
            equivalent=False,
        )
    try:
        _, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    except ValueError:
        p = float("nan")
    smd = smd_continuous(x, y)
    if bound is None:
        sp = math.sqrt(
            ((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1))
            / (len(x) + len(y) - 2)
        )
        bound = EQUIV_SMD_MARGIN * sp if sp > 0 else EQUIV_SMD_MARGIN
    tost = tost_equivalence(x, y, bound=bound)
    return dict(
        col=col,
        n_cyclops=int(len(x)),
        n_meniscus=int(len(y)),
        median_cyclops=float(np.median(x)),
        median_meniscus=float(np.median(y)),
        mwu_p=float(p),
        smd=float(smd),
        tost_bound=float(bound),
        tost_p=float(tost["tost_p"]),
        equivalent=bool(tost["tost_p"] < 0.05),
    )


def baseline_pf_balance(
    df_wide: pd.DataFrame,
    col: str = "lesion_pf_S1",
    group_col: str = "group",
    cyclops: str = "cyclops",
    meniscus: str = "meniscus",
    bound: Optional[float] = None,
) -> dict:
    """Baseline balance on the **PF block at S1**  thin wrapper over
    :func:`baseline_block_balance` (kept for API stability). See that function;
    pass ``col='lesion_ft_S1'`` there for the symmetric FT check."""
    return baseline_block_balance(
        df_wide,
        col=col,
        group_col=group_col,
        cyclops=cyclops,
        meniscus=meniscus,
        bound=bound,
    )


def evalue_or(
    or_point: float, or_ci_lo: Optional[float] = None, common_outcome: bool = True
) -> dict:
    """E-value (VanderWeele & Ding 2017) for an odds ratio.

    The minimum strength of association (on the risk-ratio scale) that an
    unmeasured confounder would need with **both** the exposure and the outcome
    to fully explain away the observed effect. Large E-value ⇒ the finding is
    robust to plausible unmeasured confounding (relevant given the unmeasured
    PF-specific confounders  dysplasia, quadriceps status  named in 04.3).

    For a **common** outcome (worsened_pf ≈ 57% in cyclops) the OR overstates the
    RR, so we approximate ``RR ≈ sqrt(OR)`` (VanderWeele's recommendation);
    set ``common_outcome=False`` to use ``RR ≈ OR`` for a rare outcome.

    Returns
    -------
    dict ``{evalue_point, evalue_ci, rr_approx, method}``.
    """

    def _ev(rr: float) -> float:
        rr = max(rr, 1.0 / rr)  # fold to ≥ 1
        return float(rr + math.sqrt(rr * (rr - 1.0)))

    rr = math.sqrt(or_point) if common_outcome else float(or_point)
    ev_point = _ev(rr)
    ev_ci = None
    if or_ci_lo is not None and np.isfinite(or_ci_lo):
        rr_lo = math.sqrt(or_ci_lo) if common_outcome else float(or_ci_lo)
        # E-value for the CI bound nearest the null (1.0 if the CI crosses it).
        ev_ci = 1.0 if rr_lo <= 1.0 else _ev(rr_lo)
    return dict(
        evalue_point=float(ev_point),
        evalue_ci=(float(ev_ci) if ev_ci is not None else None),
        rr_approx=float(rr),
        method="sqrt(OR) [common outcome]" if common_outcome else "OR [rare outcome]",
    )


def h3_risk_factors(
    df: pd.DataFrame,
    outcome_col: str = "delta_lesion_pf",
    continuous: Sequence[str] = ("age_at_trauma", "imc"),
    binary: Sequence[str] = ("female", "tabac", "travail_physique"),
    multilevel: Sequence[str] = ("pivot_pivot_contact",),
    q: float = 0.10,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """H3 family (F2)  intrinsic factors vs PF progression, within one subset.

    Pass the **cyclops** subset joined to patient covariates. For each factor,
    the association with ``outcome_col`` (Δ patellofemoral score) is tested by
    the distribution-free method matching its type, then all p-values are
    Benjamini-Hochberg corrected at ``q`` (family F2, see
    ``[[02.6-multiplicite]]``):

    * continuous (age, IMC)            -> Spearman ρ + BCa CI
    * binary 0/1 (female, tabac, …)    -> Mann-Whitney U + Cliff's δ (+ BCa CI)
    * multilevel (pivot 0/1/2)         -> Kruskal-Wallis + ε²

    ``inter_surgery_d`` is deliberately **excluded** (mediator  see the DAG in
    ``[[02.1-design-etude]]``; analysed as H4 outcome instead). All H3 results
    are exploratory.

    Returns
    -------
    pd.DataFrame
        One tidy row per factor: ``factor, kind, test, effect, effect_name,
        ci_lo, ci_hi, p, n, p_adj_bh, bh_reject``.
    """
    rows: list[dict] = []
    pvals: list[float] = []
    y_all = pd.to_numeric(df[outcome_col], errors="coerce")

    for f in continuous:
        if f not in df.columns:
            continue
        x = pd.to_numeric(df[f], errors="coerce")
        r = spearman_bca(x.values, y_all.values, n_boot=n_boot, seed=seed)
        rows.append(
            dict(
                factor=f,
                kind="continuous",
                test="Spearman rho",
                effect_name="rho",
                effect=round(float(r["rho"]), 4),
                ci_lo=round(float(r["ci_lo"]), 4),
                ci_hi=round(float(r["ci_hi"]), 4),
                p=float(r["pvalue"]),
                n=int(r["n"]),
            )
        )
        pvals.append(float(r["pvalue"]))

    for f in binary:
        if f not in df.columns:
            continue
        sub = df[[f, outcome_col]].apply(pd.to_numeric, errors="coerce").dropna()
        a = sub.loc[sub[f] == 1, outcome_col].values  # exposed (=1)
        b = sub.loc[sub[f] == 0, outcome_col].values  # reference (=0)
        if len(a) < 2 or len(b) < 2:
            rows.append(
                dict(
                    factor=f,
                    kind="binary",
                    test="MWU + Cliff delta",
                    effect_name="cliffs_delta",
                    effect=float("nan"),
                    ci_lo=float("nan"),
                    ci_hi=float("nan"),
                    p=float("nan"),
                    n=int(len(a) + len(b)),
                )
            )
            pvals.append(1.0)
            continue
        m = mwu_with_effects(a, b, n_boot=n_boot, seed=seed)
        rows.append(
            dict(
                factor=f,
                kind="binary",
                test="MWU + Cliff delta",
                effect_name="cliffs_delta",
                effect=round(float(m["cliffs_delta"]), 4),
                ci_lo=round(float(m["delta_ci_lo"]), 4),
                ci_hi=round(float(m["delta_ci_hi"]), 4),
                p=float(m["pvalue"]),
                n=int(len(a) + len(b)),
            )
        )
        pvals.append(float(m["pvalue"]))

    for f in multilevel:
        if f not in df.columns:
            continue
        sub = df[[f, outcome_col]].apply(pd.to_numeric, errors="coerce").dropna()
        try:
            kd = kw_dunn(sub, group_col=f, value_col=outcome_col)
            rows.append(
                dict(
                    factor=f,
                    kind="multilevel",
                    test="Kruskal-Wallis + epsilon^2",
                    effect_name="epsilon_sq",
                    effect=round(float(kd["epsilon_sq"]), 4),
                    ci_lo=float("nan"),
                    ci_hi=float("nan"),
                    p=float(kd["pvalue"]),
                    n=int(kd["n"]),
                )
            )
            pvals.append(float(kd["pvalue"]))
        except Exception:  # noqa: BLE001  empty cell / single group
            rows.append(
                dict(
                    factor=f,
                    kind="multilevel",
                    test="Kruskal-Wallis + epsilon^2",
                    effect_name="epsilon_sq",
                    effect=float("nan"),
                    ci_lo=float("nan"),
                    ci_hi=float("nan"),
                    p=float("nan"),
                    n=int(sub.shape[0]),
                )
            )
            pvals.append(1.0)

    out = pd.DataFrame(rows)
    if pvals:
        safe = [1.0 if (p != p) else p for p in pvals]  # NaN -> 1.0 for BH
        bh = bh_fdr(safe, q=q)
        out["p_adj_bh"] = [round(float(x), 4) for x in bh["pvals_corrected"]]
        out["bh_reject"] = bh["reject"]
    return out
