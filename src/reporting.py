"""Reporting helpers: Table 1, ArviZ summaries, markdown export.

Table 1 follows the STROBE/CONSORT convention: one row per variable, columns
per group (median [IQR] for continuous, n (%) for categorical), with a final
p-value column (MWU continuous, Fisher categorical) and SMD as descriptive
balance metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from constants import HDI_PROB, SITES


# ============================================================================
# Table 1 — baseline cohort summary
# ============================================================================

def _fmt_median_iqr(s: pd.Series) -> str:
    """Format a numeric series as ``median [Q1–Q3]`` with one decimal."""
    return f"{s.median():.1f} [{s.quantile(.25):.1f}–{s.quantile(.75):.1f}]"


def _pooled_sd(a: pd.Series, b: pd.Series) -> float:
    """Pooled SD used by Cohen's d-style SMD (descriptive only on ordinal)."""
    na, nb = len(a), len(b)
    return float(np.sqrt(
        ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    ))


def _row_continuous(var: str, a: pd.Series, b: pd.Series, g_a: str, g_b: str) -> dict:
    """One Table 1 row for a continuous variable."""
    from scipy import stats

    try:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        p = float("nan")
    sp = _pooled_sd(a, b)
    smd = (a.mean() - b.mean()) / sp if sp > 0 else float("nan")
    return {
        "variable": var, "level": "median [IQR]",
        g_a: _fmt_median_iqr(a), g_b: _fmt_median_iqr(b),
        "pvalue": f"{p:.3f}", "smd": f"{smd:+.2f}",
    }


def _smd_binary(p1: float, p2: float) -> float:
    """Austin (2009) standardised mean difference for a binary proportion.

    ``SMD = (p1 − p2) / sqrt([p1(1−p1) + p2(1−p2)] / 2)``. Reported on the
    first level's proportion so each binary covariate gets one SMD (the
    balance metric recommended by Austin for Table 1 instead of a p-value).
    """
    denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2.0)
    return float((p1 - p2) / denom) if denom > 0 else float("nan")


def _rows_categorical(
    var: str, df_patient: pd.DataFrame, sa: pd.DataFrame, sb: pd.DataFrame,
    g_a: str, g_b: str, group_col: str,
) -> list[dict]:
    """List of Table 1 rows for one categorical variable (one row per level).

    SMD column (Austin) is populated for binary variables on the reference
    (first) level; multi-level variables leave SMD blank.
    """
    from scipy import stats

    levels = sorted(df_patient[var].dropna().unique())
    smd_str = ""
    if len(levels) == 2:
        tab = np.array([
            [(sa[var] == levels[0]).sum(), (sa[var] == levels[1]).sum()],
            [(sb[var] == levels[0]).sum(), (sb[var] == levels[1]).sum()],
        ])
        _, p = stats.fisher_exact(tab)
        # Austin SMD on the proportion of the *second* level (the "event").
        na_b, nb_b = len(sa), len(sb)
        p1 = (sa[var] == levels[1]).sum() / na_b if na_b else float("nan")
        p2 = (sb[var] == levels[1]).sum() / nb_b if nb_b else float("nan")
        smd_str = f"{_smd_binary(p1, p2):+.2f}"
    else:
        try:
            tab = pd.crosstab(df_patient[group_col], df_patient[var])
            _, p, _, _ = stats.chi2_contingency(tab.values)
        except Exception:  # noqa: BLE001
            p = float("nan")
    out: list[dict] = []
    for lvl in levels:
        na = int((sa[var] == lvl).sum())
        nb = int((sb[var] == lvl).sum())
        out.append({
            "variable": var, "level": str(lvl),
            g_a: _fmt_n_pct(na, len(sa)),
            g_b: _fmt_n_pct(nb, len(sb)),
            "pvalue": f"{p:.3f}" if lvl == levels[0] else "",
            "smd": smd_str if lvl == levels[0] else "",
        })
    return out


def _fmt_n_pct(n: int, denom: int) -> str:
    """Format ``n (xx.x%)`` safely; returns ``—`` if denom == 0."""
    if denom <= 0:
        return "—"
    return f"{n} ({100 * n / denom:.1f}%)"


def make_table1(
    df_patient: pd.DataFrame,
    continuous: Iterable[str] = (
        "age_at_trauma", "imc", "taille", "poids",
    ),
    categorical: Iterable[str] = (
        "sexe", "pivot_pivot_contact", "travail_physique", "tabac",
    ),
    group_col: str = "group",
) -> pd.DataFrame:
    """Build a STROBE-style Table 1.

    Parameters
    ----------
    df_patient : pd.DataFrame
        One row per patient (output of :func:`preprocessing.to_patient`).
    continuous : iterable of str
        Continuous variables → median [Q1–Q3] + MWU p-value + SMD.
    categorical : iterable of str
        Categorical variables → n (%) per level + Fisher exact p-value.
    group_col : str

    Returns
    -------
    pd.DataFrame
        Tidy table with columns: ``variable, level, <group_a>, <group_b>,
        pvalue, smd``.
    """
    groups = sorted(df_patient[group_col].dropna().unique())
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups, got {groups}")
    g_a, g_b = groups
    sa = df_patient[df_patient[group_col] == g_a]
    sb = df_patient[df_patient[group_col] == g_b]

    rows: list[dict] = [
        {"variable": "n", "level": "",
         g_a: str(len(sa)), g_b: str(len(sb)),
         "pvalue": "", "smd": ""}
    ]

    for var in continuous:
        if var not in df_patient.columns:
            continue
        a = pd.to_numeric(sa[var], errors="coerce").dropna()
        b = pd.to_numeric(sb[var], errors="coerce").dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        rows.append(_row_continuous(var, a, b, g_a, g_b))

    for var in categorical:
        if var not in df_patient.columns:
            continue
        rows.extend(_rows_categorical(var, df_patient, sa, sb, g_a, g_b, group_col))

    return pd.DataFrame(rows)


# ============================================================================
# ArviZ summary wrapper (uses modern ci_prob / ci_kind API)
# ============================================================================

def summary_bayes(
    idata,
    var_names: Optional[Iterable[str]] = None,
    hdi_prob: float = HDI_PROB,
) -> pd.DataFrame:
    """Posterior summary with HDI + P(>0).

    Uses the modern ArviZ API (``ci_prob`` / ``ci_kind``) when available and
    falls back to the legacy ``hdi_prob`` keyword otherwise. Adds a ``P(>0)``
    column (posterior tail probability) for directional inference.

    Parameters
    ----------
    idata : az.InferenceData
    var_names : optional iterable
    hdi_prob : float, default 0.94 (contract: HDI 94%)
    """
    import arviz as az
    import inspect

    sig = inspect.signature(az.summary)
    kwargs = dict(var_names=list(var_names) if var_names else None)
    if "ci_prob" in sig.parameters:
        kwargs.update(ci_prob=hdi_prob, ci_kind="hdi")
    else:
        kwargs.update(hdi_prob=hdi_prob, kind="stats")
    summ = az.summary(idata, **kwargs)

    # P(>0) per variable
    p_gt_0 = {}
    posterior = idata.posterior
    for v in (var_names or list(posterior.data_vars)):
        if v not in posterior.data_vars:
            continue
        vals = posterior[v].values.ravel()
        p_gt_0_full = float((vals > 0).mean())
        # If multi-dim, per-coord P(>0) handled below
        p_gt_0[v] = p_gt_0_full

    # Add P(>0) column by matching the index name root
    def _p_gt(name: str) -> float:
        root = name.split("[")[0]
        if root not in posterior.data_vars:
            return float("nan")
        arr = posterior[root]
        if arr.ndim == 2:  # chain × draw scalar
            return float((arr.values > 0).mean())
        # multi-dim: parse coord from name e.g. "beta_c[trochlée]"
        if "[" not in name:
            return float((arr.values > 0).mean())
        coord_val = name[name.index("[") + 1:name.rindex("]")]
        # Pick last dim (excluding chain/draw). Guard for scalar posteriors
        # whose dims are {chain, draw} only — defensive: return overall P(>0).
        extra_dims = [d for d in arr.dims if d not in ("chain", "draw")]
        if not extra_dims:
            return float((arr.values > 0).mean())
        last_dim = extra_dims[-1]
        sub = arr.sel({last_dim: coord_val}) if coord_val in arr[last_dim].values else arr
        return float((sub.values > 0).mean())

    summ["P(>0)"] = [_p_gt(n) for n in summ.index]
    return summ


# ============================================================================
# Markdown export
# ============================================================================

def export_to_markdown(
    table_df: pd.DataFrame, path: Path | str, title: str = "",
) -> Path:
    """Write a DataFrame as a markdown table (suitable for Obsidian).

    Parameters
    ----------
    table_df : pd.DataFrame
    path : Path or str
        Destination .md file. Parent dirs created if missing.
    title : str
        Optional H2 heading prepended.

    Returns
    -------
    Path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = table_df.to_markdown(index=False)
    content = (f"## {title}\n\n{body}\n") if title else (body + "\n")
    path.write_text(content, encoding="utf-8")
    return path


# ============================================================================
# Auto-narrative formatters (interpretability)
# ============================================================================

def format_test_result(res: dict, test_name: str = "mwu") -> str:
    """Translate a frequentist test dict into a 1-2 sentence French narrative.

    Parameters
    ----------
    res : dict
        Output of ``tests_freq.mwu_with_effects``, ``wilcoxon_exact_with_rrb``,
        ``mcnemar_exact_midp``, ``fisher_exact_2x2``, ``spearman_bca``,
        or ``kw_dunn``.
    test_name : str
        One of ``"mwu"``, ``"wilcoxon"``, ``"mcnemar"``, ``"fisher"``,
        ``"spearman"``, ``"kw"``.

    Returns
    -------
    str
        A pre-formatted sentence stating statistic, p-value, effect size
        with verbal magnitude, CI, and a one-clause clinical interpretation.
    """
    t = test_name.lower()
    p = res.get("pvalue", float("nan"))
    sig = "significatif" if p < 0.05 else "NS"
    if t == "mwu":
        U = res["statistic"]
        d = res["cliffs_delta"]
        lo, hi = res["delta_ci_lo"], res["delta_ci_hi"]
        mag = res.get("cliffs_delta_magnitude", "n/a")
        ps = res.get("probability_of_superiority", float("nan"))
        return (
            f"Mann-Whitney U = {U:.1f} (p = {p:.3f}, {sig}). "
            f"Cliff's δ = {d:+.2f} ({mag}, IC 95% [{lo:+.2f}, {hi:+.2f}]), "
            f"P(X > Y) = {ps:.2f}."
        )
    if t == "wilcoxon":
        W = res["statistic"]
        r = res["rank_biserial"]
        mag = res.get("rrb_magnitude", "n/a")
        npr = res.get("n_pairs", "?")
        return (
            f"Wilcoxon signed-rank W = {W:.1f} sur {npr} paires "
            f"(p = {p:.3f}, {sig}). r_rb = {r:+.2f} ({mag})."
        )
    if t == "mcnemar":
        b, c = res.get("b", "?"), res.get("c", "?")
        midp = res.get("mid_p", float("nan"))
        return (
            f"McNemar exact p = {p:.3f} (mid-p = {midp:.3f}, {sig}); "
            f"discordants b={b}, c={c}."
        )
    if t == "fisher":
        odds = res.get("odds_ratio", float("nan"))
        midp = res.get("mid_p", float("nan"))
        return (
            f"Fisher exact p = {p:.3f} (mid-p = {midp:.3f}, {sig}); "
            f"OR = {odds:.2f}."
        )
    if t == "spearman":
        rho = res["rho"]
        lo, hi = res["ci_lo"], res["ci_hi"]
        n = res.get("n", "?")
        return (
            f"Spearman ρ = {rho:+.2f} (IC 95% [{lo:+.2f}, {hi:+.2f}], "
            f"n = {n}, p = {p:.3f}, {sig})."
        )
    if t == "kw":
        H = res["statistic"]
        eps = res.get("epsilon_sq", float("nan"))
        return (
            f"Kruskal-Wallis H = {H:.2f} (p = {p:.3f}, {sig}); "
            f"ε² = {eps:.3f}."
        )
    return repr(res)


# ============================================================================
# English verdict phrases (consensus point G — single Bayesian decision rule)
# ============================================================================

def verdict_bayes_en(
    idata,
    var: str,
    label: str,
    threshold: float = 0.95,
    hdi_prob: float = HDI_PROB,
    direction: str = "greater",
    two_sided: bool = True,
) -> dict:
    """Automatic English verdict for a Bayesian estimand (point G amended 2026-05-29).

    Two decision regimes, because using a **one-sided** P(effect>0)≥threshold
    rule on an effect whose **direction was suggested by the data** double-counts
    the data (the Bayesian analogue of choosing a one-sided test after seeing the
    sign — revue-methodo 2026-05-29):

    * ``two_sided=True`` (**default — for POST-HOC estimands** such as the
      patellofemoral contrast ``delta_pf`` / ``contrast_pf_ft``): the estimand is
      "supported" iff the ``hdi_prob`` **HDI excludes 0** (a direction-agnostic
      credible rule); ``P(effect>0)`` is reported only **descriptively**::

        "PF contrast supported: 94% HDI [+1.42, +4.05] excludes 0
         (two-sided credible rule; P(effect>0)=99.8%, descriptive)."

    * ``two_sided=False`` (**for a PRE-SPECIFIED directional estimand** — e.g. the
      global knee-wide δ̄ under the directional H1 "cyclops worsen cartilage"):
      the legitimate one-sided rule ``P(effect>0)≥threshold`` (0.95 primary /
      0.90 secondary).

    Parameters
    ----------
    idata : az.InferenceData
    var : str
        Scalar posterior variable name (e.g. ``"delta_pf"``, ``"delta_bar"``).
    label : str
        Human label for the estimand.
    threshold : float, default 0.95
        One-sided posterior threshold (used only when ``two_sided=False``).
    hdi_prob : float, default 0.94
    direction : {"greater", "less"}
        Tested direction (used only when ``two_sided=False``).
    two_sided : bool, default True
        Decision regime (see above). Default True is the safe choice for any
        estimand whose direction was not pre-registered.

    Returns
    -------
    dict
        ``{var, label, rule, p_gt0, p_direction, threshold, supported,
        hdi_lo, hdi_hi, hdi_excludes_0, sentence}`` (keys stable across regimes;
        regime-irrelevant fields are still present for a uniform shape).
    """
    vals = np.asarray(idata.posterior[var].values).ravel()
    p_gt = float((vals > 0).mean())
    hdi_lo, hdi_hi = _hdi_from_samples(vals, hdi_prob)
    excludes0 = bool((hdi_lo > 0) or (hdi_hi < 0))

    if two_sided:
        supported = excludes0
        status = "supported" if supported else "not supported"
        sentence = (
            f"{label} {status}: {int(hdi_prob * 100)}% HDI "
            f"[{hdi_lo:+.2f}, {hdi_hi:+.2f}] "
            f"{'excludes' if excludes0 else 'includes'} 0 "
            f"(two-sided credible rule; P(effect>0) = {p_gt:.1%}, descriptive)."
        )
        return dict(
            var=var, label=label, rule="two_sided",
            p_gt0=p_gt, p_direction=max(p_gt, 1 - p_gt), threshold=None,
            supported=supported, hdi_lo=hdi_lo, hdi_hi=hdi_hi,
            hdi_excludes_0=excludes0, sentence=sentence,
        )

    p = p_gt if direction == "greater" else (1.0 - p_gt)
    sym = ">" if direction == "greater" else "<"
    supported = bool(p >= threshold)
    status = "supported" if supported else "not supported"
    sentence = (
        f"{label} {status} (pre-specified directional): "
        f"P({var} {sym} 0 | data) = {p:.1%} ({int(threshold * 100)}% threshold). "
        f"{int(hdi_prob * 100)}% HDI [{hdi_lo:+.2f}, {hdi_hi:+.2f}]."
    )
    return dict(
        var=var, label=label, rule="one_sided_prespecified",
        p_gt0=p_gt, p_direction=p, threshold=threshold,
        supported=supported, hdi_lo=hdi_lo, hdi_hi=hdi_hi,
        hdi_excludes_0=excludes0, sentence=sentence,
    )


def confidence_phrase_en(p: float, claim: str) -> str:
    """\"we are X% confident that <claim>\" — point-G narrative helper."""
    return f"we are {p:.0%} confident that {claim}"


def verdict_freq_en(
    pvalue: float,
    label: str,
    bh_q: Optional[float] = None,
    bh_reject: Optional[bool] = None,
    effect: Optional[str] = None,
) -> str:
    """English verdict for a frequentist test, optionally with a BH-FDR clause.

    Generates sentences such as::

        "PF contrast: permutation p = 0.0002 (large effect, Cliff δ = +0.53)."
        "sum-6 progression: not significant after BH-FDR q = 0.10."

    Parameters
    ----------
    pvalue : float
    label : str
        Estimand label.
    bh_q : float, optional
        BH-FDR target; if given the BH clause is appended.
    bh_reject : bool, optional
        Whether the hypothesis survived BH-FDR at ``bh_q``.
    effect : str, optional
        Effect-size clause to append (e.g. ``"Cliff δ = +0.53 (large)"``).
    """
    parts = [f"{label}: p = {pvalue:.4g}"]
    if effect:
        parts[0] += f" ({effect})"
    if bh_q is not None and bh_reject is not None:
        if bh_reject:
            parts.append(f"significant after BH-FDR q = {bh_q:g}")
        else:
            parts.append(f"not significant after BH-FDR q = {bh_q:g}")
    return ". ".join(parts) + "."


def _hdi_from_samples(vals: np.ndarray, hdi_prob: float) -> tuple[float, float]:
    """Smallest interval covering ``hdi_prob`` of the sorted samples.

    Avoids relying on the ArviZ ``az.hdi`` keyword name (which has changed
    multiple times: ``hdi_prob`` → ``ci_prob`` → positional ``prob``).
    """
    s = np.sort(np.asarray(vals).ravel())
    n = len(s)
    if n == 0:
        return float("nan"), float("nan")
    k = max(1, int(np.floor(hdi_prob * n)))
    widths = s[k - 1:] - s[: n - k + 1]
    i = int(np.argmin(widths))
    return float(s[i]), float(s[i + k - 1])


def interpret_bayes(
    idata,
    var: str,
    rope: tuple[float, float] = (-0.1, 0.1),
    hdi_prob: float = HDI_PROB,
) -> dict:
    """Bayesian narrative: posterior mean, HDI, P(direction), ROPE %, verdict.

    Parameters
    ----------
    idata : az.InferenceData
    var : str
        Posterior variable name (scalar; for indexed vars, pass e.g.
        ``"beta_c"`` and the function returns per-coord stats stacked).
    rope : (float, float)
        Region of practical equivalence on the latent scale. Default ±0.1
        is appropriate for a logit-scale slope; pick wider for raw outcomes.
    hdi_prob : float
        HDI mass (default 0.94, Kruschke convention).

    Returns
    -------
    dict
        ``{mean, hdi_lo, hdi_hi, p_direction, rope_pct, hdi_excludes_rope,
        narrative}``. ``p_direction`` = max(P(β > 0), P(β < 0)). ROPE % is
        the % of posterior mass inside ``rope``. ``hdi_excludes_rope`` is
        True iff the HDI lies entirely outside ``rope``.
    """
    arr = idata.posterior[var]
    vals = np.asarray(arr.values).ravel()
    mean = float(vals.mean())
    p_above = float((vals > 0).mean())
    p_dir = max(p_above, 1 - p_above)
    rope_lo, rope_hi = float(rope[0]), float(rope[1])
    rope_pct = float(((vals >= rope_lo) & (vals <= rope_hi)).mean() * 100.0)
    hdi_lo, hdi_hi = _hdi_from_samples(vals, hdi_prob)

    excludes = (hdi_hi < rope_lo) or (hdi_lo > rope_hi)
    direction = "↑" if p_above >= 0.5 else "↓"
    verdict = (
        "effet probable hors équivalence pratique"
        if excludes and p_dir >= 0.95
        else "effet probable" if p_dir >= 0.95
        else "effet incertain" if p_dir >= 0.80
        else "absence d'effet plausible"
    )
    narrative = (
        f"{var} = {mean:+.2f} (HDI {int(hdi_prob*100)}% "
        f"[{hdi_lo:+.2f}, {hdi_hi:+.2f}]), pd = {p_dir:.2%} {direction}, "
        f"ROPE% = {rope_pct:.1f}% → {verdict}."
    )
    return dict(
        mean=mean, hdi_lo=hdi_lo, hdi_hi=hdi_hi,
        p_direction=p_dir, rope_pct=rope_pct,
        hdi_excludes_rope=bool(excludes),
        narrative=narrative,
    )
