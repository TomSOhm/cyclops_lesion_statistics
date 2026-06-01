"""PyMC models M1-M5 for the cyclops vs méniscus study.

| ID | Type                              | Outcome                  | Method              |
|----|-----------------------------------|--------------------------|---------------------|
| M1 | Beta-binomial conjugate           | worsened_pf              | analytical          |
| M2 | Neg-binomial (SANITY ONLY)        | PF score S2 | S1 (no Δ⁺) | NUTS via bambi      |
| M3 | Heterogeneous hierarchical        | y[i,t,c] (mixed)         | NUTS (PRIMARY ⭐⭐)  |
| M4 | Weibull AFT                       | inter_surgery_d          | lifelines MLE       |
| M5 | LogNormal                         | inter_surgery_d (H4)     | NUTS                |

M3 refactor (revue-methodo 2026-05 + 2026-05-29 amendment, consensus points C/E)
--------------------------------------------------------------------------------
* Scale COLLAPSED to {0, 1, ≥2} (point A) before modelling.
* HETEROGENEOUS likelihood (E1, amended): Bernoulli/logit for pte, pti, cfe, cfi
  (≤ 2 events ≥2 each → upper cutpoint prior-driven); OrderedLogistic {0,1,≥2}
  (2 free cutpoints) only for trochlée & rotule.
* SELECTABLE pooling (E2 generalised, point C): both β_c (time slope) and δ_c
  (Group×Time interaction) are partially pooled toward group means whose grouping
  is set by ``pooling`` — "exchangeable" (one mean = knee-wide δ̄, the selection-
  immune primary), "two_block" (PF/FT) or "three_cluster". The topographic
  structure is TESTED by LOO (:func:`compare_pooling_structures`), never assumed.
* δ_c per compartment ~ Normal(μ_group[grouping(c)], σ_δ); ``delta_comp`` is always
  exposed. exchangeable → ``delta_bar``; two_block → ``delta_pf`` / ``delta_ft`` /
  ``contrast_pf_ft`` (a candidate-structure output, NOT a pre-specified co-primary).
  The PF localisation is read as a DERIVED contrast from the exchangeable posterior
  (:func:`derived_pf_contrast`) — non-circular.
* FREE cutpoints per ordinal compartment (E3) — never shared (measurement ≠ effect).
* η = β_c[comp]·t + γ·g + δ_c[comp]·t·g + u[patient].
* t ∈ {−0.5, +0.5} is the S2−S1 CONTRAST, **not** a per-year slope (E5).
* Student-t(3) priors on effects (E4); tightened HalfNormal on σ.

NUTS defaults (contract §3 + §7, locked)::

    chains=4, tune=2000, draws=2000, target_accept=0.95, random_seed=42

Convergence thresholds::

    r_hat ≤ 1.01, ess_bulk ≥ 400, divergences == 0

Escalation if non-convergent: ``target_accept=0.99`` + ``tune=4000``
(parameterisation is already non-centred). Log decision in
``99-Project-Log/decisions.md``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from constants import (
    BLOCKS,
    ESS_MIN,
    HDI_PROB,
    NUTS_KWARGS,
    NUTS_KWARGS_ESCALATED,
    RANDOM_SEED,
    RHAT_MAX,
    SCORE_MAX,
    SCORE_MAX_COLLAPSED,
    SCORE_MIN,
    SITE_BLOCK,
    SITES,
    SITES_BINARY,
    SITES_FT,
    SITES_ORDINAL,
    SITES_PF,
)


# ============================================================================
# M1 — Beta-Binomial conjugate (analytical)
# ============================================================================

def fit_m1_beta_binomial(
    k: int, N: int, alpha: float = 1.0, beta: float = 1.0, hdi_prob: float = 0.94,
) -> dict:
    """Beta-binomial conjugate posterior for P(worsening).

    Prior :math:`\\theta \\sim \\text{Beta}(\\alpha, \\beta)` and likelihood
    :math:`k \\mid \\theta \\sim \\text{Binomial}(N, \\theta)` give the closed
    form posterior :math:`\\theta \\mid k, N \\sim \\text{Beta}(\\alpha+k,
    \\beta+N-k)`.

    Parameters
    ----------
    k : int
        Successes (e.g. number of patients with ``worsened_any == 1``).
    N : int
        Trials (total patients in group).
    alpha, beta : float, default 1.0
        Beta prior parameters (Jeffreys default = 0.5; uniform = 1.0).
    hdi_prob : float
        HDI mass (contract: 0.94).

    Returns
    -------
    dict
        ``{alpha_post, beta_post, mean, mode, hdi_lo, hdi_hi}``.
    """
    from scipy import stats as sps

    a_post = alpha + k
    b_post = beta + N - k
    mean = a_post / (a_post + b_post)
    if a_post > 1 and b_post > 1:
        mode = (a_post - 1) / (a_post + b_post - 2)
    else:
        mode = float("nan")
    lo = (1 - hdi_prob) / 2
    hi = 1 - lo
    ci_lo, ci_hi = sps.beta.ppf([lo, hi], a_post, b_post)
    return dict(
        alpha_post=a_post, beta_post=b_post, mean=float(mean), mode=float(mode),
        hdi_lo=float(ci_lo), hdi_hi=float(ci_hi), hdi_prob=hdi_prob,
        k=int(k), N=int(N), alpha=alpha, beta=beta,
    )


# ============================================================================
# M2 — NegBin SANITY-CHECK on PF score S2 | S1 (NOT primary; no Δ⁺ truncation)
# ============================================================================

def fit_m2_negbin_sanity(
    df_wide: pd.DataFrame,
    outcome_col: str = "lesion_pf_S2",
    baseline_col: str = "lesion_pf_S1",
    formula: Optional[str] = None,
    **sample_kwargs,
):
    """NegBin/Poisson sanity-check on the PF score at S2 adjusted for S1.

    .. warning::
       **Demoted from the primary inferential chain** (consensus point D).
       The earlier M2 summed non-commensurable ordinals *and* truncated
       improvements via :math:`\\Delta^+ = \\max(\\Delta, 0)`, which discards
       information and is doctrinally incoherent. The truncation is
       **removed**. This function is retained only as a labelled descriptive
       sanity-check: a count regression of the post-operative PF score
       (``lesion_pf_S2`` ∈ {0..4}, already ≥ 0 — **no shift, no truncation**)
       on the group, adjusting for the baseline PF score (``lesion_pf_S1``).
       Primary inference is M3 + the permutation test.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Output of :func:`preprocessing.to_wide` joined with covariates.
    outcome_col : str
        Non-negative count outcome (default post-op PF score).
    baseline_col : str
        Baseline adjustment covariate (default pre-op PF score).
    formula : str, optional
        Override the Wilkinson formula. Defaults to
        ``"{outcome} ~ 1 + group + {baseline}"``.
    sample_kwargs : dict
        Override ``NUTS_KWARGS`` (e.g. ``draws=500`` for smoke test).

    Returns
    -------
    dict ``{model, idata, outcome_col, baseline_col, note}``.
    """
    import bambi as bmb

    if formula is None:
        formula = f"{outcome_col} ~ 1 + group + {baseline_col}"

    df = df_wide.copy()
    needed = [c for c in (outcome_col, baseline_col, "group") if c in df.columns]
    df = df[needed].dropna().copy()
    # Outcome is a non-negative cartilage score → no shift / no Δ⁺ truncation.
    df[outcome_col] = pd.to_numeric(df[outcome_col]).astype(int)

    model = bmb.Model(formula, data=df, family="negativebinomial")
    kwargs = {**NUTS_KWARGS, **sample_kwargs}
    idata = model.fit(**kwargs)
    return dict(
        model=model, idata=idata,
        outcome_col=outcome_col, baseline_col=baseline_col,
        note="SANITY-CHECK only (point D): no Δ⁺ truncation; not primary.",
    )


# ============================================================================
# M3 — Hierarchical ordinal cumulative logit ⭐⭐ PRIMARY INFERENTIAL
# ============================================================================

def _melt_long_long(
    df_long: pd.DataFrame,
    sites: Sequence[str],
    id_col: str,
    group_col: str,
    collapse: bool = True,
) -> pd.DataFrame:
    """Melt the (patient × surgery) frame into a (patient × time × compartment) frame.

    Encodes the indices and design variables consumed by the M3 PyMC model
    (consensus point E):

    * ``patient_idx`` — composite ``(group, anonyme)`` factorisation (so the
      19 reused ids are never merged).
    * ``comp_idx`` — compartment code (fixed ``sites`` order).
    * ``block_idx`` — topographic block, 0 = PF, 1 = FT (point B/E2).
    * ``is_binary`` — True for PTI / CFI (Bernoulli sites, point E1).
    * ``t`` — **centred time contrast** ∈ {−0.5 (S1), +0.5 (S2)} (point E5);
      β_c·t is algebraically the S2−S1 contrast, *not* a per-year slope.
    * ``g`` — group indicator (0 = meniscus, 1 = cyclops).

    Parameters
    ----------
    df_long : pd.DataFrame
        One row per (patient × surgery) with the site columns.
    sites, id_col, group_col : see callers.
    collapse : bool, default True
        Apply the {0,1,≥2} collapse (grade 3 → 2) before indexing
        (point A). Set False only for native-scale diagnostics.

    Returns
    -------
    pd.DataFrame
        Long-long frame, NaN scores dropped.
    """
    import preprocessing as _pp

    sites = list(sites)
    src = _pp.collapse_scores(df_long, sites) if collapse else df_long
    long = src.melt(
        id_vars=[id_col, group_col, "surgery_num"],
        value_vars=sites, var_name="comp", value_name="y",
    ).dropna(subset=["y"])
    # Coerce nullable Int64 → numpy int, absorbing any residual pd.NA.
    y_num = pd.to_numeric(long["y"], errors="coerce")
    if y_num.isna().any():
        long = long.loc[y_num.notna()].copy()
        y_num = y_num.loc[y_num.notna()]
    long["y"] = y_num.astype(int).values
    upper = SCORE_MAX_COLLAPSED if collapse else SCORE_MAX
    assert long["y"].between(SCORE_MIN, upper).all(), (
        f"y outside [{SCORE_MIN}, {upper}] after melt — check input scoring/collapse."
    )
    long["patient_idx"] = pd.factorize(
        long[group_col].astype(str) + "_" + long[id_col].astype(str)
    )[0]
    long["comp_idx"] = pd.Categorical(long["comp"], categories=sites).codes
    long["block_idx"] = long["comp"].map(SITE_BLOCK).astype(int)
    long["is_binary"] = long["comp"].isin(SITES_BINARY)
    # Centred time contrast (point E5): S1 → −0.5, S2 → +0.5.
    long["t"] = np.where(long["surgery_num"] == 2, 0.5, -0.5)
    long["g"] = (long[group_col] == "cyclops").astype(int)
    return long


def _pooling_grouping(pooling: str, sites: Sequence[str]):
    """Map each compartment to a pooling-group index for the chosen structure.

    * ``"exchangeable"`` — one group ``["all"]`` (fully exchangeable 6 sites);
      the group mean is the knee-wide δ̄, the **selection-immune global estimand**.
    * ``"two_block"``    — ``["PF", "FT"]`` (the topographic partition; now a
      *candidate structure* compared by LOO, not an assumed hypothesis).
    * ``"three_cluster"``— ``["PF", "FT_antlat", "FT_med"]`` = {trochlée,rotule} /
      {pte,cfe} / {pti,cfi}, the 2–3 clusters the Δ-correlations actually show
      (PTE–CFE ≈ 0.70, PTI–CFI ≈ 0.39; revue-methodo 2026-05-29).

    Returns ``(group_names, group_of_comp)`` where ``group_of_comp[i]`` is the
    group index of ``sites[i]``.
    """
    sites = list(sites)
    if pooling == "exchangeable":
        idx = {s: 0 for s in sites}
        names = ["all"]
    elif pooling == "two_block":
        names = list(BLOCKS)  # ["PF", "FT"]
        idx = {s: SITE_BLOCK[s] for s in sites}
    elif pooling == "three_cluster":
        names = ["PF", "FT_antlat", "FT_med"]
        cl = {"trochlée": 0, "rotule": 0, "pte": 1, "cfe": 1, "pti": 2, "cfi": 2}
        idx = {s: cl[s] for s in sites}
    else:
        raise ValueError(
            f"pooling must be exchangeable|two_block|three_cluster, got {pooling!r}"
        )
    return names, np.array([idx[s] for s in sites], dtype=int)


def _build_m3_model(
    long: pd.DataFrame,
    sites: Sequence[str],
    score_min: int = SCORE_MIN,
    score_max: int = SCORE_MAX_COLLAPSED,
    prior_sigma_beta: float = 1.0,
    prior_sigma_delta: float = 1.0,
    pooling: str = "two_block",
):
    """Construct (without fitting) the refactored heterogeneous M3 model.

    Implements consensus point E in full. Centralises the spec so that
    :func:`fit_m3`, :func:`fit_m3_with_prior` and :func:`ppc_m3` share an
    identical likelihood.

    Structure
    ---------
    For observation i (patient × time × compartment c, collapsed scale)::

        η_i = β_c[comp_i]·t_i + γ·g_i + δ_block[block_i]·t_i·g_i + u[pat_i]

    with t ∈ {−0.5, +0.5} (S2−S1 contrast, **not** a per-year slope).

    * **Heterogeneous likelihood (E1, amended 2026-05-29).** Ordinal sites
      (trochlée, rotule) use ``pm.OrderedLogistic`` on {0,1,≥2} with **free**
      per-compartment cutpoints (2 each). Binary sites (pte, pti, cfe, cfi) use
      ``pm.Bernoulli`` with ``p = sigmoid(η)`` — pte/cfe reclassed to binary as
      their grade-≥2 cutpoint rests on only 2 events. Both likelihood blocks
      share β_c, γ, δ_c and u.
    * **Selectable pooling on β_c and δ_c (E2 generalised, point C).** Both the
      time slope β_c and the interaction δ_c are partially pooled toward group
      means whose grouping is set by ``pooling`` (``exchangeable`` → one mean =
      knee-wide; ``two_block`` → PF/FT; ``three_cluster``). Pooling acts on the
      *effect*, never on the measurement.
    * **Estimands.** ``exchangeable`` exposes ``delta_bar`` (selection-immune
      global knee-wide interaction); ``two_block`` exposes ``delta_pf``,
      ``delta_ft``, ``contrast_pf_ft`` — now a *candidate structure* compared by
      LOO (:func:`compare_pooling_structures`), not an assumed co-primary.
      ``delta_comp`` (per-compartment δ) is always exposed for the derived PF
      contrast (:func:`derived_pf_contrast`).
    * **Free cutpoints per ordinal compartment (E3).** One ordered
      ``Normal`` 2-vector per ordinal site (no shared ``cut``).
    * **Student-t(3) priors on effects (E4)**, tightened HalfNormal on σ.

    Parameters
    ----------
    long : pd.DataFrame
        Output of :func:`_melt_long_long` (collapsed scale).
    sites : sequence of str
        Compartment labels (coord values, fixed order).
    score_min, score_max : int
        Collapsed bounds (0..2).
    prior_sigma_beta : float, default 1.0
        Scale of the HalfNormal prior on ``sigma_beta``.

    Returns
    -------
    pymc.Model
    """
    import pymc as pm
    import pytensor.tensor as pt

    sites = list(sites)
    n_patients = long["patient_idx"].nunique()
    n_cuts = score_max - score_min          # = 2 on the collapsed scale
    ordinal_sites = [s for s in sites if s in SITES_ORDINAL]
    binary_sites = [s for s in sites if s in SITES_BINARY]

    # Pooling-group index per compartment (revue 2026-05-29, point C): both the
    # time slope β_c and the Group×Time interaction δ_c are partially pooled
    # toward a group mean whose grouping is SELECTABLE, so the topographic
    # structure is TESTED (LOO across structures) rather than assumed.
    # "exchangeable" is the selection-immune primary.
    pgroup_names, group_of_comp = _pooling_grouping(pooling, sites)

    coords = {
        "patient": np.arange(n_patients),
        "comp": sites,
        "pgroup": pgroup_names,
        "cutpoint_idx": np.arange(n_cuts),
        "ord_comp": ordinal_sites,
    }

    cut_init = np.linspace(-1.0, 1.0, n_cuts)

    # Pre-split observation indices by likelihood family.
    is_bin = long["is_binary"].values
    ord_mask = ~is_bin
    bin_mask = is_bin

    with pm.Model(coords=coords) as model:
        # --- Time-slope pooling β_c over pooling groups (E2 generalised) ---
        mu_beta = pm.StudentT("mu_beta", nu=3, mu=0.0, sigma=1.0, dims="pgroup")
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=prior_sigma_beta)
        beta_offset = pm.StudentT("beta_offset", nu=3, mu=0.0, sigma=1.0, dims="comp")
        beta_c = pm.Deterministic(
            "beta_c", mu_beta[group_of_comp] + sigma_beta * beta_offset, dims="comp",
        )

        # --- Group main effect ---------------------------------------------
        gamma = pm.StudentT("gamma", nu=3, mu=0.0, sigma=1.0)

        # --- Per-compartment Group×Time interaction δ_c, pooled to group ----
        # δ_c ~ Normal(μ_group[grouping(c)], σ_δ) (non-centred). The group MEANS
        # are the estimands; with one group this is the knee-wide δ̄. Per-comp
        # δ_c is always exposed for the derived PF contrast (point C).
        mu_delta = pm.StudentT("mu_delta", nu=3, mu=0.0, sigma=1.0, dims="pgroup")
        sigma_delta = pm.HalfNormal("sigma_delta", sigma=prior_sigma_delta)
        delta_offset = pm.Normal("delta_offset", mu=0.0, sigma=1.0, dims="comp")
        delta_comp = pm.Deterministic(
            "delta_comp", mu_delta[group_of_comp] + sigma_delta * delta_offset,
            dims="comp",
        )

        # Named estimands per structure.
        if pooling == "exchangeable":
            pm.Deterministic("delta_bar", mu_delta[0])          # knee-wide global
        elif pooling == "two_block":
            pm.Deterministic("delta_pf", mu_delta[0])
            pm.Deterministic("delta_ft", mu_delta[1])
            pm.Deterministic("contrast_pf_ft", mu_delta[0] - mu_delta[1])

        # --- Patient random intercept (non-centred, retained — E5) --------
        sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
        u_offset = pm.Normal("u_offset", mu=0.0, sigma=1.0, dims="patient")
        u = pm.Deterministic("u", sigma_u * u_offset, dims="patient")

        # --- Linear predictor η for every observation ---------------------
        comp_idx = long["comp_idx"].values
        pat_idx = long["patient_idx"].values
        t = long["t"].values.astype(float)
        g = long["g"].values.astype(float)

        eta = (
            beta_c[comp_idx] * t
            + gamma * g
            + delta_comp[comp_idx] * t * g
            + u[pat_idx]
        )

        # --- Ordinal likelihood block (free cutpoints per compartment) ----
        if ordinal_sites:
            # One ordered 2-vector of cutpoints per ordinal compartment.
            cut = pm.Normal(
                "cut",
                mu=cut_init,
                sigma=2.0,
                transform=pm.distributions.transforms.ordered,
                # No explicit initval: mu=cut_init ([-1, 1], already increasing)
                # is a valid ordered moment, and a non-default initial_value
                # breaks pm.compute_log_likelihood for external NUTS (nutpie) —
                # which we need for the LOO pooling-structure comparison.
                dims=("ord_comp", "cutpoint_idx"),
            )
            # Map each ordinal observation to its position within `ordinal_sites`.
            ord_code = {s: k for k, s in enumerate(ordinal_sites)}
            ord_pos_full = long["comp"].map(ord_code)
            ord_idx = ord_pos_full[ord_mask].astype(int).values
            eta_ord = eta[ord_mask]
            y_ord = long["y"].values[ord_mask].astype(int)
            pm.OrderedLogistic(
                "y_ord",
                eta=eta_ord,
                cutpoints=cut[ord_idx],          # per-obs cutpoint row
                observed=y_ord,
                compute_p=False,
            )

        # --- Binary likelihood block (PTI / CFI) --------------------------
        if binary_sites:
            eta_bin = eta[bin_mask]
            y_bin = (long["y"].values[bin_mask] > 0).astype(int)
            pm.Bernoulli(
                "y_bin",
                logit_p=eta_bin,
                observed=y_bin,
            )
    return model


def fit_m3(
    df_long: pd.DataFrame,
    sites: Sequence[str] = tuple(SITES),
    id_col: str = "anonyme",
    group_col: str = "group",
    score_min: int = SCORE_MIN,
    score_max: int = SCORE_MAX_COLLAPSED,
    nuts_sampler: str = "nutpie",
    pooling: str = "two_block",
    prior_sigma_delta: float = 1.0,
    compute_log_likelihood: bool = False,
    **sample_kwargs,
):
    """Fit the refactored heterogeneous hierarchical M3 (PRIMARY model).

    Heterogeneous likelihood + two-block pooling + free cutpoints +
    per-block Group×Time interaction (consensus point E; see
    :func:`_build_m3_model`). The collapsed {0,1,≥2} scale is applied inside
    :func:`_melt_long_long`. Returns ``InferenceData`` with the named
    estimands ``delta_pf``, ``delta_ft`` and ``contrast_pf_ft``.

    Parameters
    ----------
    df_long : pd.DataFrame
        Long format (one row per patient × surgery), with ``surgery_num``,
        ``group``, ``id_col`` and the site columns.
    sites : sequence of str
        Compartment columns (default: SITES — all six).
    score_min, score_max : int
        Collapsed bounds (0..2).
    nuts_sampler : str, default "nutpie"
        PyMC ``nuts_sampler`` backend. ``nutpie`` (numba) is used because the
        default pytensor C backend has no compiler available on this host
        (g++ missing on the Python 3.14 conda env); ``nutpie`` JIT-compiles
        via numba and samples fine. Pass ``"pymc"`` to force the slow fallback.
    sample_kwargs : dict
        Override :data:`NUTS_KWARGS`.

    Returns
    -------
    az.InferenceData
    """
    import pymc as pm

    long = _melt_long_long(df_long, sites, id_col, group_col, collapse=True)
    model = _build_m3_model(
        long, sites, score_min=score_min, score_max=score_max,
        prior_sigma_delta=prior_sigma_delta, pooling=pooling,
    )
    with model:
        kwargs = {**NUTS_KWARGS, **sample_kwargs}
        if compute_log_likelihood:
            kwargs["idata_kwargs"] = {
                **kwargs.get("idata_kwargs", {}), "log_likelihood": True,
            }
        idata = pm.sample(nuts_sampler=nuts_sampler, **kwargs)
    return idata


# Backwards-compatible alias (old name used by notebooks / earlier callers).
fit_m3_hierarchical_ordinal = fit_m3


def fit_m3_with_prior(
    df_long: pd.DataFrame,
    prior_sigma_beta: float = 1.0,
    sites: Sequence[str] = tuple(SITES),
    id_col: str = "anonyme",
    group_col: str = "group",
    score_min: int = SCORE_MIN,
    score_max: int = SCORE_MAX_COLLAPSED,
    nuts_sampler: str = "nutpie",
    **sample_kwargs,
):
    """Fit M3 with a configurable HalfNormal scale on ``sigma_beta``.

    Sensitivity wrapper. Run with ``prior_sigma_beta`` in
    :data:`SENSITIVITY_PRIOR_SIGMAS` (= ``(0.5, 1.0, 5.0)``) and compare the
    posterior of ``delta_pf`` / ``contrast_pf_ft`` to gauge the influence of
    the hierarchical prior tightness.

    Parameters
    ----------
    df_long : pd.DataFrame
        Same input as :func:`fit_m3`.
    prior_sigma_beta : float, default 1.0
        Scale of ``sigma_beta`` HalfNormal prior. Larger ⇒ less shrinkage.

    Returns
    -------
    az.InferenceData
    """
    import pymc as pm

    long = _melt_long_long(df_long, sites, id_col, group_col, collapse=True)
    model = _build_m3_model(
        long, sites,
        score_min=score_min, score_max=score_max,
        prior_sigma_beta=prior_sigma_beta,
    )
    with model:
        kwargs = {**NUTS_KWARGS, **sample_kwargs}
        idata = pm.sample(nuts_sampler=nuts_sampler, **kwargs)
    return idata


def derived_pf_contrast(idata, hdi_prob: float = HDI_PROB) -> dict:
    """PF-FT contrast DERIVED from the per-compartment δ_c posterior (point C).

    The honest, non-circular reading of the topographic localisation: take the
    joint posterior of ``delta_comp`` (per-compartment Group×Time interaction)
    from a model that imposed NO PF/FT partition (fit with
    ``pooling='exchangeable'``) and form (mean δ over PF sites) − (mean δ over
    FT sites). Reported as the pattern the data REVEAL, with the explicit caveat
    that the partition was selected (so the amplitude is an optimistic bound).

    Returns
    -------
    dict ``{mean, hdi_lo, hdi_hi, p_gt0, pf_sites, ft_sites, hdi_prob}``.
    """
    da = idata.posterior["delta_comp"]
    comp_dim = [d for d in da.dims if d not in ("chain", "draw")][0]
    comps = [str(c) for c in da[comp_dim].values]
    pf = [c for c in comps if c in SITES_PF]
    ft = [c for c in comps if c in SITES_FT]
    pf_mean = da.sel({comp_dim: pf}).mean(comp_dim)
    ft_mean = da.sel({comp_dim: ft}).mean(comp_dim)
    contrast = (pf_mean - ft_mean).values.ravel()
    s = np.sort(contrast)
    n = len(s)
    k = max(1, int(np.floor(hdi_prob * n)))
    w = s[k - 1:] - s[: n - k + 1]
    i = int(np.argmin(w))
    return dict(
        mean=float(contrast.mean()), hdi_lo=float(s[i]), hdi_hi=float(s[i + k - 1]),
        p_gt0=float((contrast > 0).mean()), pf_sites=pf, ft_sites=ft,
        hdi_prob=hdi_prob,
    )


def _combine_loglik(idata):
    """Concatenate y_ord + y_bin pointwise log-likelihoods into one ``obs`` dim.

    The heterogeneous M3 has two observed RVs over disjoint observation subsets;
    :func:`arviz.loo` needs a single pointwise log-likelihood vector. We stack
    each RV's non-sample dims into ``obs`` and concatenate → one
    ``(chain, draw, obs)`` array, so LOO / compare operate on all
    (patient×time×compartment) cells jointly.
    """
    import arviz as az
    import xarray as xr

    ll = idata.log_likelihood
    parts = []
    for v in ll.data_vars:
        da = ll[v]
        obs_dims = [d for d in da.dims if d not in ("chain", "draw")]
        if obs_dims:
            st = da.stack(obs=obs_dims).reset_index("obs", drop=True)
        else:
            st = da.expand_dims("obs")
        parts.append(st.transpose("chain", "draw", "obs"))
    combined = xr.concat(parts, dim="obs").transpose("chain", "draw", "obs")
    # arviz 1.1.0 InferenceData is a DataTree; build via from_dict({group: ...}).
    # az.loo needs a posterior group present (to derive r_eff), so carry the real
    # posterior across alongside the combined pointwise log-likelihood "y".
    post_ds = idata.posterior
    return az.from_dict({
        "posterior": {v: post_ds[v].values for v in post_ds.data_vars},
        "log_likelihood": {"y": combined.values},
    })


def compare_pooling_structures(
    df_long: pd.DataFrame,
    sites: Sequence[str] = tuple(SITES),
    id_col: str = "anonyme",
    group_col: str = "group",
    poolings: Sequence[str] = ("exchangeable", "two_block", "three_cluster"),
    nuts_sampler: str = "nutpie",
    **sample_kwargs,
) -> dict:
    """Fit M3 under each pooling structure and rank by LOO (consensus point C).

    Makes the topographic organisation an empirical RESULT rather than an
    assumption: if LOO favours ``two_block`` / ``three_cluster`` over
    ``exchangeable``, the data REJECT 6-site exchangeability and support a block
    structure — dissolving the circularity of *assuming* PF/FT. At n=69 the
    comparison may be inconclusive (flat ELPD); report the ELPD differences and
    their SE and say so rather than over-claiming.

    Returns
    -------
    dict
        ``{idatas, loos, compare}`` — ``idatas`` keyed by pooling,
        ``loos`` the per-model :func:`arviz.loo`, ``compare`` the
        :func:`arviz.compare` table (ranked by ELPD-LOO).
    """
    import arviz as az

    idatas, combined = {}, {}
    for pool in poolings:
        idata = fit_m3(
            df_long, sites=sites, id_col=id_col, group_col=group_col,
            nuts_sampler=nuts_sampler, pooling=pool,
            compute_log_likelihood=True, **sample_kwargs,
        )
        idatas[pool] = idata
        combined[pool] = _combine_loglik(idata)
    loos = {k: az.loo(v, var_name="y") for k, v in combined.items()}
    cmp = az.compare(combined, var_name="y")
    return dict(idatas=idatas, loos=loos, compare=cmp)


def ppc_m3(
    idata,
    df_long: pd.DataFrame,
    n_draws: int = 500,
    sites: Sequence[str] = tuple(SITES),
    id_col: str = "anonyme",
    group_col: str = "group",
    score_min: int = SCORE_MIN,
    score_max: int = SCORE_MAX_COLLAPSED,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """Posterior predictive check for the heterogeneous M3.

    Samples posterior-predictive replicates of **both** likelihood blocks
    (``y_ord`` ordinal + ``y_bin`` binary), stitches them back into the full
    (patient × time × compartment) layout, and computes predicted category
    frequencies grouped by (group, time, compartment) alongside the empirical
    frequencies. The collapsed {0,1,≥2} scale is used throughout.

    Parameters
    ----------
    idata : az.InferenceData
        Output of :func:`fit_m3` / :func:`fit_m3_with_prior`.
    df_long : pd.DataFrame
        Same input data used to fit ``idata``.
    n_draws : int, default 500
        Posterior draws used for the PPC summary.

    Returns
    -------
    dict
        ``{idata_ppc, observed_freq, predicted_freq_mean, predicted_freq_hdi,
        n_categories}`` — each ``*_freq`` is long-form
        (group, time, comp, y, freq).
    """
    import pymc as pm

    long = _melt_long_long(df_long, sites, id_col, group_col, collapse=True)
    model = _build_m3_model(long, sites, score_min=score_min, score_max=score_max)

    ord_mask = (~long["is_binary"]).values
    bin_mask = long["is_binary"].values

    rng = np.random.default_rng(random_seed)
    var_names = [v for v in ("y_ord", "y_bin")
                 if v in [rv.name for rv in model.observed_RVs]]
    with model:
        idata_ppc = pm.sample_posterior_predictive(
            idata, var_names=var_names, random_seed=rng,
        )

    # Reassemble per-observation predictions into the full long order.
    def _stack(name: str) -> np.ndarray:
        da = idata_ppc.posterior_predictive[name]
        return da.stack(sample=("chain", "draw")).transpose("sample", ...).values

    n_obs = len(long)
    parts = {}
    if "y_ord" in var_names:
        parts["ord"] = _stack("y_ord")
    if "y_bin" in var_names:
        parts["bin"] = _stack("y_bin")
    n_samp = next(iter(parts.values())).shape[0]
    yrep = np.empty((n_samp, n_obs), dtype=int)
    if "ord" in parts:
        yrep[:, ord_mask] = parts["ord"]
    if "bin" in parts:
        yrep[:, bin_mask] = parts["bin"]

    obs = long.assign(
        time=np.where(long["t"].values > 0, "S2", "S1"),
        group=long[group_col].values,
    ).reset_index(drop=True)

    observed_freq = (
        obs.groupby(["group", "time", "comp", "y"]).size()
        .rename("count").reset_index()
    )
    observed_freq["freq"] = (
        observed_freq.groupby(["group", "time", "comp"])["count"]
        .transform(lambda s: s / s.sum())
    )

    rows = []
    for d in range(min(n_draws, n_samp)):
        tmp = obs.assign(y_pred=yrep[d])
        freq = (
            tmp.groupby(["group", "time", "comp", "y_pred"]).size()
            .rename("count").reset_index()
        )
        freq["freq"] = (
            freq.groupby(["group", "time", "comp"])["count"]
            .transform(lambda s: s / s.sum())
        )
        freq["draw"] = d
        rows.append(freq.rename(columns={"y_pred": "y"}))
    pred = pd.concat(rows, ignore_index=True)

    pred_mean = (
        pred.groupby(["group", "time", "comp", "y"])["freq"].mean()
        .rename("freq_mean").reset_index()
    )
    pred_hdi = (
        pred.groupby(["group", "time", "comp", "y"])["freq"]
        .agg(lo=lambda s: float(np.quantile(s, 0.03)),
             hi=lambda s: float(np.quantile(s, 0.97)))
        .reset_index()
    )

    return dict(
        idata_ppc=idata_ppc,
        observed_freq=observed_freq,
        predicted_freq_mean=pred_mean,
        predicted_freq_hdi=pred_hdi,
        n_categories=score_max - score_min + 1,
    )


# ============================================================================
# M4 — Weibull AFT on inter_surgery_d
# ============================================================================

def fit_m4_weibull_aft(
    df_wide: pd.DataFrame,
    duration_col: str = "inter_surgery_d",
    covariates: Sequence[str] = ("group", "imc", "pivot_pivot_contact"),
) -> dict:
    """Weibull Accelerated Failure Time on inter-surgery delay.

    No censoring → ``event_observed = 1`` everywhere.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Must contain ``duration_col`` and the covariates.
    covariates : sequence of str

    Returns
    -------
    dict with fitted lifelines model and summary frame.
    """
    from lifelines import WeibullAFTFitter

    sub = df_wide[[duration_col, *covariates]].copy()
    # Encode categorical group → 0/1.
    if "group" in sub.columns and sub["group"].dtype == object:
        sub["group"] = (sub["group"] == "cyclops").astype(int)
    # lifelines requires plain float64 — it chokes on pandas nullable Int64
    # (inter_surgery_d) and object columns ("Values must be numeric").
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    # Drop covariates with NO usable values (e.g. ``imc`` is not recorded in this
    # cohort) — otherwise a full-NaN column makes ``dropna`` empty the frame.
    usable_cov = [c for c in covariates if sub[c].notna().any()]
    sub = sub[[duration_col, *usable_cov]].dropna().astype(float)
    sub["event"] = 1.0

    aft = WeibullAFTFitter()
    aft.fit(sub, duration_col=duration_col, event_col="event")
    return dict(model=aft, summary=aft.summary, covariates_used=usable_cov)


# ============================================================================
# M5 — LogNormal on a delay variable
# ============================================================================

def fit_m5_lognormal(
    df: pd.DataFrame,
    var: str = "trauma_to_surgery_d",
    group_col: str = "group",
    **sample_kwargs,
):
    """LogNormal posterior per group on a positive delay variable.

    Estimates :math:`\\mu_g, \\sigma_g` for each group separately.

    Returns
    -------
    az.InferenceData
    """
    import pymc as pm

    sub = df[[group_col, var]].dropna()
    sub = sub[sub[var] > 0]
    groups = sub[group_col].astype("category")
    g_idx = groups.cat.codes.values
    g_names = list(groups.cat.categories)

    coords = {"group": g_names, "obs": np.arange(len(sub))}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", mu=np.log(sub[var].median()), sigma=2.0, dims="group")
        sigma = pm.HalfNormal("sigma", sigma=2.0, dims="group")
        pm.LogNormal("y", mu=mu[g_idx], sigma=sigma[g_idx],
                     observed=sub[var].values, dims="obs")
        kwargs = {**NUTS_KWARGS, **sample_kwargs}
        idata = pm.sample(**kwargs)
    return idata


# ============================================================================
# Sensitivity / structural checks for M3
# ============================================================================

def cutpoint_summary(idata, var_name: str = "cut") -> dict:
    """Posterior summary of the per-compartment free cutpoints.

    The refactored M3 uses **free** cutpoints per ordinal compartment
    (consensus point E3 — measurement is never pooled). The old
    proportional-odds gap check is therefore not the relevant diagnostic;
    instead we report, per ordinal compartment, the posterior mean cutpoints
    and the inter-cutpoint gap ``cut[·,1] − cut[·,0]`` (which encodes how the
    {0→1} and {1→≥2} thresholds separate for that site).

    Parameters
    ----------
    idata : az.InferenceData
        Fitted M3 idata containing ``cut`` with dims (ord_comp, cutpoint_idx).
    var_name : str, default ``"cut"``

    Returns
    -------
    dict
        ``{per_comp: {comp: {cut_means, gap_mean, gap_std}}, n_cutpoints}``
        or ``{error: ...}`` if ``cut`` is absent (e.g. an all-binary subset).
    """
    if var_name not in idata.posterior:
        return dict(error=f"{var_name!r} absent (no ordinal compartments in this fit).")

    post = idata.posterior[var_name]  # (chain, draw, ord_comp, cutpoint_idx)
    dims = [d for d in post.dims if d not in ("chain", "draw")]
    if "cutpoint_idx" not in post.dims:
        return dict(error="cut has no cutpoint_idx dim.")
    comp_dim = next((d for d in dims if d != "cutpoint_idx"), None)
    flat = post.stack(sample=("chain", "draw"))  # (..., sample)

    per_comp = {}
    comps = list(post[comp_dim].values) if comp_dim else [None]
    for c in comps:
        sub = flat.sel({comp_dim: c}) if comp_dim else flat
        vals = sub.values  # (cutpoint_idx, sample)
        if vals.shape[0] < 2:
            continue
        gap = vals[1] - vals[0]
        per_comp[str(c)] = dict(
            cut_means=[float(v) for v in vals.mean(axis=1)],
            gap_mean=float(gap.mean()),
            gap_std=float(gap.std()),
        )
    return dict(per_comp=per_comp, n_cutpoints=int(post.sizes["cutpoint_idx"]))


# Backwards-compatible alias (older callers referenced this name).
proportional_odds_check = cutpoint_summary


def loo_compare(
    idata_m2,
    idata_m3,
    model_names: Sequence[str] = ("M2_negbin", "M3_hierordinal"),
) -> pd.DataFrame:
    """LOO-CV comparison between M2 (NegBin) and M3 (hierarchical ordinal).

    Wraps ``az.compare`` after computing :func:`az.loo` on each ``InferenceData``.
    Returns the ArviZ comparison table (sorted by ELPD).

    Notes
    -----
    LOO comparison across different likelihoods/data is **descriptive**, not
    a formal inferential test — the two models use different outcomes
    (Δlesion_total shifted for M2, ordinal per-compartment for M3) and the
    ELPD scales are not directly comparable. We report it as in
    [[02.5-modeles-bayesiens]] §6.3 with the explicit caveat that selection
    should be guided by inferential goal (M3 is primary for H2 regardless).

    Parameters
    ----------
    idata_m2, idata_m3 : az.InferenceData
        Must contain ``log_likelihood`` (set ``idata_kwargs={'log_likelihood':
        True}`` when sampling, or use ``az.loo(idata, pointwise=True)``).
    model_names : sequence of 2 str

    Returns
    -------
    pd.DataFrame
        Output of ``az.compare`` (rank, elpd_loo, p_loo, weight, etc.).
    """
    import arviz as az

    n_a, n_b = model_names[0], model_names[1]
    compare_dict = {n_a: idata_m2, n_b: idata_m3}
    cmp = az.compare(compare_dict, ic="loo", scale="log")
    return cmp


# ============================================================================
# Convergence diagnostics
# ============================================================================

def check_convergence(idata, rhat_max: float = RHAT_MAX, ess_min: float = ESS_MIN) -> dict:
    """Assert r_hat ≤ ``rhat_max`` and ess_bulk ≥ ``ess_min`` for all params.

    Parameters
    ----------
    idata : az.InferenceData
    rhat_max : float, default 1.01
    ess_min : float, default 400

    Returns
    -------
    dict
        ``{ok, max_rhat, min_ess_bulk, n_divergent, offenders}``.
    """
    import arviz as az

    rhat = az.rhat(idata)
    try:
        ess = az.ess(idata, method="bulk")
    except TypeError:
        try:
            ess = az.ess(idata, kind="bulk")
        except TypeError:
            ess = az.ess(idata)

    max_rhat = float(max(np.nanmax(rhat[v].values) for v in rhat.data_vars))
    min_ess = float(min(np.nanmin(ess[v].values) for v in ess.data_vars))
    groups = idata.groups() if callable(idata.groups) else idata.groups
    n_div = int(idata.sample_stats["diverging"].sum()) if "sample_stats" in groups else 0

    offenders = []
    for v in rhat.data_vars:
        if np.nanmax(rhat[v].values) > rhat_max:
            offenders.append((v, "rhat", float(np.nanmax(rhat[v].values))))
        if np.nanmin(ess[v].values) < ess_min:
            offenders.append((v, "ess_bulk", float(np.nanmin(ess[v].values))))

    ok = (max_rhat <= rhat_max) and (min_ess >= ess_min) and (n_div == 0)
    return dict(ok=bool(ok), max_rhat=max_rhat, min_ess_bulk=min_ess,
                n_divergent=n_div, offenders=offenders,
                thresholds=dict(rhat_max=rhat_max, ess_min=ess_min))


def fit_m3_with_escalation(
    df_long: pd.DataFrame,
    sites: Sequence[str] = tuple(SITES),
    id_col: str = "anonyme",
    group_col: str = "group",
    nuts_sampler: str = "nutpie",
    pooling: str = "two_block",
    compute_log_likelihood: bool = False,
    rhat_max: float = RHAT_MAX,
    ess_min: float = ESS_MIN,
    **sample_kwargs,
) -> dict:
    """Fit M3 and auto-escalate NUTS if convergence fails (consensus point I.2).

    Runs :func:`fit_m3` with :data:`NUTS_KWARGS`; if :func:`check_convergence`
    reports r̂ > ``rhat_max``, ESS < ``ess_min`` or divergences > 0, refits once
    with :data:`NUTS_KWARGS_ESCALATED` (``target_accept=0.99``, ``tune=4000``;
    the parameterisation is already non-centred). Returns both the (possibly
    escalated) ``idata`` and the convergence reports so ``run_all`` can print
    the escalation trail.

    Returns
    -------
    dict
        ``{idata, convergence, escalated, convergence_initial}``.
    """
    base = {k: v for k, v in NUTS_KWARGS.items()}
    base.update(sample_kwargs)
    idata = fit_m3(df_long, sites=sites, id_col=id_col, group_col=group_col,
                   nuts_sampler=nuts_sampler, pooling=pooling,
                   compute_log_likelihood=compute_log_likelihood, **base)
    conv0 = check_convergence(idata, rhat_max=rhat_max, ess_min=ess_min)
    escalated = False
    if not conv0["ok"]:
        escalated = True
        esc = {k: v for k, v in NUTS_KWARGS_ESCALATED.items()}
        esc.update(sample_kwargs)
        idata = fit_m3(df_long, sites=sites, id_col=id_col, group_col=group_col,
                       nuts_sampler=nuts_sampler, pooling=pooling,
                       compute_log_likelihood=compute_log_likelihood, **esc)
        conv = check_convergence(idata, rhat_max=rhat_max, ess_min=ess_min)
    else:
        conv = conv0
    return dict(idata=idata, convergence=conv, escalated=escalated,
                convergence_initial=conv0)


# ============================================================================
# Clinician-facing posterior risk extractor (plan item I10)
# ============================================================================

def posterior_risk_at_S2(
    ppc_out: dict,
    group: str,
    score: int = SCORE_MAX_COLLAPSED,
    time: str = "S2",
    comp: Optional[str] = None,
    hdi_prob: float = HDI_PROB,
) -> dict:
    """Posterior predictive probability P(y = score | group, time, [comp]).

    Extracts an absolute-risk number from M3's posterior predictive output,
    aggregated either to a single compartment (``comp`` given) or marginal
    across all compartments (``comp=None``, returning the average per-group
    risk).

    Parameters
    ----------
    ppc_out : dict
        Output of :func:`ppc_m3` (must contain ``pred_mean`` and ``pred_hdi``,
        or equivalently the long-form keys ``predicted_freq_mean`` /
        ``predicted_freq_hdi`` returned by the current implementation).
    group : str
        Group label (e.g. ``"cyclops"`` or ``"meniscus"``).
    score : int
        Ordinal score value (default ``SCORE_MAX`` = 2, "complete loss").
    time : str
        ``"S1"`` or ``"S2"`` (default S2 — the post-second-surgery state).
    comp : str, optional
        Compartment label; if ``None`` averages across all compartments.
    hdi_prob : float
        Mass of the credible band reported (currently used only for the
        narrative label; the HDI bounds returned are whatever ``pred_hdi``
        recorded, which by default is a 94% band).

    Returns
    -------
    dict
        ``{group, time, comp, score, p_mean, p_lo, p_hi, narrative}`` —
        ``narrative`` is a French one-liner suitable for a clinical summary.
    """
    pred_mean = ppc_out.get("pred_mean", ppc_out.get("predicted_freq_mean"))
    pred_hdi = ppc_out.get("pred_hdi", ppc_out.get("predicted_freq_hdi"))

    sel_mean = pred_mean[
        (pred_mean["group"] == group)
        & (pred_mean["time"] == time)
        & (pred_mean["y"] == score)
    ]
    sel_hdi = pred_hdi[
        (pred_hdi["group"] == group)
        & (pred_hdi["time"] == time)
        & (pred_hdi["y"] == score)
    ]
    if comp is not None:
        sel_mean = sel_mean[sel_mean["comp"] == comp]
        sel_hdi = sel_hdi[sel_hdi["comp"] == comp]

    if sel_mean.empty:
        return dict(
            group=group, time=time, comp=comp, score=score,
            p_mean=float("nan"), p_lo=float("nan"), p_hi=float("nan"),
            narrative=(
                f"P(y={score} | {group}, {time}, comp={comp}) introuvable "
                f"dans pred_mean."
            ),
        )

    p_mean = float(sel_mean["freq_mean"].mean())
    if not sel_hdi.empty:
        p_lo = float(sel_hdi["lo"].mean())
        p_hi = float(sel_hdi["hi"].mean())
    else:
        p_lo, p_hi = float("nan"), float("nan")

    label_score = {0: "aucune", 1: "partielle", 2: "complète"}.get(score, str(score))
    comp_str = comp if comp else "toutes compartiments"
    narrative = (
        f"P(lésion {label_score} à {time} | {group}, {comp_str}) = "
        f"{p_mean:.1%} [HDI {int(hdi_prob*100)}% {p_lo:.1%}, {p_hi:.1%}]."
    )
    return dict(
        group=group, time=time, comp=comp, score=score,
        p_mean=p_mean, p_lo=p_lo, p_hi=p_hi,
        narrative=narrative,
    )
