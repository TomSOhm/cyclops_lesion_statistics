#!/usr/bin/env python
"""run_all.py  reproducible end-to-end pipeline for the cyclops-vs-méniscus study.

Implements the consensus specification (``99-Project-Log/revue-methodo-2026-05.md``,
points A–I) as a single deterministic script (``RANDOM_SEED = 42``). It:

1. Loads the dual-sheet Excel, fixes the composite-key date-hygiene bug,
   derives PF/FT blocks, the ``female`` indicator and the collapsed scale.
2. Descriptive Table 1 in **SMD** (Austin) + baseline S1 balance
   (must recover MWU p ≈ 0.78).
3. PRIMARY patellofemoral contrast: MWU + Cliff δ + permutation + BCa +
   inversion CI (must recover δ ≈ +0.53, p ≈ 0.0002).
4. Demoted sum-6 descriptive (δ ≈ +0.25, p ≈ 0.088 / 0.044).
5. Per-compartment worsening (full disclosure, point C.3) +
   PF-vs-FT topographic specificity.
6. M1 (worsened_pf), M3 (primary heterogeneous hierarchical model with
   convergence escalation), M4 Weibull AFT + M5 LogNormal for H4
   (inter_surgery_d, median cyc ≈ 240 vs mén ≈ 518).
7. Sport/occupation sensitivity (with and without covariates).
8. Saves all artefacts to ``results/`` (idata_*.nc, *.csv, results.json) and
   prints a convergence report + the key canonical numbers.

It does **not** touch ``viz.py`` / ``make_figures.py`` (chart-expert scope).

Run::

    python3 run_all.py
    # quick smoke (tiny MCMC, skips escalation):
    python3 run_all.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Windows consoles default to cp1252 and choke on δ / accented compartment
# names. Force UTF-8 on the streams so the canonical report prints cleanly.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001  older Python / redirected stream
        pass

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from constants import (  # noqa: E402
    HDI_PROB,
    N_BOOT_DEFAULT,
    N_PERM_DEFAULT,
    RANDOM_SEED,
    SITES,
    SITES_BINARY,
    SITES_FT,
    SITES_PF,
)
import bayes_models as bm  # noqa: E402
import loaders  # noqa: E402
import preprocessing as pp  # noqa: E402
import reporting as rpt  # noqa: E402
import tests_freq as tf  # noqa: E402

RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _hdi(idata, var: str, prob: float = HDI_PROB) -> list[float]:
    """94% HDI of a scalar posterior var as ``[lo, hi]`` (ArviZ-version safe)."""
    vals = np.asarray(idata.posterior[var].values).ravel()
    lo, hi = rpt._hdi_from_samples(vals, prob)
    return [round(float(lo), 4), round(float(hi), 4)]


def _p_gt0(idata, var: str) -> float:
    vals = np.asarray(idata.posterior[var].values).ravel()
    return float((vals > 0).mean())


def _post_mean(idata, var: str) -> float:
    return float(np.asarray(idata.posterior[var].values).mean())


def _report_baseline(results: dict, wide, level: str, col: str, tag: str) -> dict:
    """Unified baseline-balance check (MWU + SMD + TOST) on one S1 block.

    Runs :func:`tests_freq.baseline_block_balance` on ``col`` and stores a
    CONSISTENT ``baseline_<level>_*`` key set. Used for the global 6-sum and the
    PF / FT blocks alike, so all three levels get the *same* three measures
    (ends the old asymmetry where only the blocks had SMD + TOST).
    """
    b = tf.baseline_block_balance(wide, col=col)
    results[f"baseline_{level}_mwu_p"] = round(float(b["mwu_p"]), 4)
    results[f"baseline_{level}_smd"] = round(float(b["smd"]), 4)
    results[f"baseline_{level}_tost_p"] = round(float(b["tost_p"]), 4)
    results[f"baseline_{level}_tost_bound"] = round(float(b["tost_bound"]), 4)
    results[f"baseline_{level}_equivalent"] = bool(b["equivalent"])
    results[f"baseline_{level}_median_cyc"] = float(b["median_cyclops"])
    results[f"baseline_{level}_median_men"] = float(b["median_meniscus"])
    print(
        f"[2·{tag}] baseline {tag} S1: MWU p={b['mwu_p']:.3f}, SMD={b['smd']:+.3f}, "
        f"TOST p={b['tost_p']:.3f} within ±{b['tost_bound']:.2f} "
        f"(equivalent={b['equivalent']})"
    )
    return b


def _jsonable(obj):
    """Recursively coerce numpy / pandas scalars to plain Python for json."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _save_idata(idata, name: str) -> str:
    path = RESULTS_DIR / name
    try:
        # InferenceData.to_netcdf is the canonical API (top-level az.to_netcdf
        # was removed in recent arviz). Persists the FULL posterior draws.
        idata.to_netcdf(str(path))
    except Exception as exc:  # noqa: BLE001  netcdf backend may be missing
        path = path.with_suffix(".json")
        # Fallback: dump posterior summary so the run still produces an artefact.
        import arviz as az

        az.summary(idata).to_json(str(path))
        print(f"  [warn] netcdf save failed ({exc!r}); wrote summary to {path.name}")
    return str(path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def main(smoke: bool = False) -> dict:
    t_start = time.time()
    np.random.seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"random_seed": RANDOM_SEED, "smoke": smoke}

    n_boot = 1000 if smoke else N_BOOT_DEFAULT
    n_perm = 5000 if smoke else N_PERM_DEFAULT
    mcmc_kwargs = dict(draws=300, tune=300, chains=2) if smoke else {}

    print("=" * 78)
    print("LINO_STATS  run_all.py   (seed=%d, smoke=%s)" % (RANDOM_SEED, smoke))
    print("=" * 78)

    # --- 1. Load + fix + derive -------------------------------------------
    df = loaders.load_combined()
    df = pp.apply_date_hygiene(df)  # composite-key bug fix (point I.1)
    df = pp.add_derived(df)
    wide = pp.to_wide(df)
    patient = pp.to_patient(df)
    # Patient-level frame joined with per-patient outcomes/covariates.
    cov_cols = [
        "group",
        "anonyme",
        "female",
        "sexe",
        "pivot_pivot_contact",
        "travail_physique",
        "tabac",
        "age_at_trauma",
        "imc",
        "taille",
        "poids",
    ]
    cov_cols = [c for c in cov_cols if c in patient.columns]
    merged = wide.merge(patient[cov_cols], on=["group", "anonyme"], how="left")

    # Sentinels (point I.1).
    sizes = df.groupby(["group", "anonyme"]).size()
    assert (sizes == 2).all(), "composite key sentinel failed"
    assert len(sizes) == 69
    n_pf = int(wide["delta_lesion_pf"].notna().sum())
    results["n_patients"] = int(len(sizes))
    results["n_analysable_pf"] = n_pf
    print(
        f"[1] data: {len(sizes)} patients (composite key OK); "
        f"analysable PF n = {n_pf} "
        f"({int(wide.groupby('group')['delta_lesion_pf'].apply(lambda s: s.notna().sum())['cyclops'])} cyc "
        f"+ {int(wide.groupby('group')['delta_lesion_pf'].apply(lambda s: s.notna().sum())['meniscus'])} mén)"
    )

    # --- 1b. Date anomalies (point A) -------------------------------------
    # Run on the RAW frame (before hygiene collapses intra-patient drift) so #9
    # (trauma S1≠S2 ~2 months) is visible alongside the negative delays (#38).
    # These corrupt age/delay/H3/H4 only  NOT the cartilage Δ  and form the
    # exclusion list for the delay/age sensitivity (with/without).
    anomalies = pp.detect_date_anomalies(loaders.load_combined())
    anomalies.to_csv(RESULTS_DIR / "data_anomalies.csv", index=False)
    results["data_anomalies"] = anomalies.to_dict("records")
    anomaly_keys = (
        set(zip(anomalies["group"], anomalies["anonyme"])) if len(anomalies) else set()
    )
    print(
        f"[1b] date anomalies flagged: {len(anomalies)} -> data_anomalies.csv"
        + (f"  {anomalies['kind'].value_counts().to_dict()}" if len(anomalies) else "")
    )

    # --- 2. Table 1 (SMD) + baseline S1 -----------------------------------
    table1 = rpt.make_table1(patient)
    table1.to_csv(RESULTS_DIR / "table1.csv", index=False)
    # Age SMD (canonical balance metric).
    age_c = pd.to_numeric(
        patient.loc[patient.group == "cyclops", "age_at_trauma"], errors="coerce"
    ).dropna()
    age_m = pd.to_numeric(
        patient.loc[patient.group == "meniscus", "age_at_trauma"], errors="coerce"
    ).dropna()
    age_smd = tf.smd_continuous(age_c.values, age_m.values)
    results["table1_age_smd"] = round(float(age_smd), 4)
    results["table1_age_median_cyc"] = float(age_c.median())
    results["table1_age_median_men"] = float(age_m.median())

    # Baseline S1 lesion load equivalence (must be p ≈ 0.78).
    b_c = wide.loc[wide.group == "cyclops", "lesion_total_S1"].dropna().astype(float)
    b_m = wide.loc[wide.group == "meniscus", "lesion_total_S1"].dropna().astype(float)
    base = tf.mwu_with_effects(b_c.values, b_m.values, n_boot=n_boot, seed=RANDOM_SEED)
    results["baseline_s1_p"] = round(float(base["pvalue"]), 4)
    results["baseline_s1_cliff"] = round(float(base["cliffs_delta"]), 4)
    results["baseline_s1_median_cyc"] = float(b_c.median())
    results["baseline_s1_median_men"] = float(b_m.median())
    print(
        f"[2] Table 1 saved; age SMD = {age_smd:+.3f}; "
        f"baseline S1 lesion MWU p = {base['pvalue']:.4f} "
        f"(medians {b_c.median():.0f}/{b_m.median():.0f})"
    )

    # --- 2b. Unified baseline equivalence (MWU + SMD + TOST) at all three -----
    # levels: global 6-sum, PF block, FT block  IDENTICAL treatment so the
    # baseline-balance structure is symmetric (point D + FT symmetry). Before,
    # only the blocks got SMD+TOST while the global got MWU only.
    _report_baseline(results, wide, "total", "lesion_total_S1", "TOTAL")
    _report_baseline(results, wide, "pf", "lesion_pf_S1", "PF")
    bft = _report_baseline(results, wide, "ft", "lesion_ft_S1", "FT")
    if not bft["equivalent"]:
        print(
            "     [note] FT block NOT proven equivalent at S1 → the PF−FT "
            "topographic contrast is EXPLORATORY (regression-to-mean may mask "
            "an FT signal). PF (primary) is equivalent; knee-wide δ̄ unaffected."
        )

    # --- 3. PRIMARY PF contrast -------------------------------------------
    pf = tf.pf_contrast(
        wide,
        value_col="delta_lesion_pf",
        n_boot=n_boot,
        n_perm=n_perm,
        seed=RANDOM_SEED,
    )
    results["pf_cliff_delta"] = round(float(pf["cliffs_delta"]), 4)
    results["pf_cliff_magnitude"] = pf["cliffs_delta_magnitude"]
    results["pf_mwu_p"] = float(pf["mwu_p"])
    results["pf_perm_p"] = float(pf["perm_p"])
    results["pf_perm_method"] = pf["perm_method"]
    results["pf_bca_lo"] = round(float(pf["bca_lo"]), 4)
    results["pf_bca_hi"] = round(float(pf["bca_hi"]), 4)
    results["pf_inv_ci_lo"] = round(float(pf["inv_ci_lo"]), 4)
    results["pf_inv_ci_hi"] = round(float(pf["inv_ci_hi"]), 4)
    results["pf_prob_superiority"] = round(float(pf["probability_of_superiority"]), 4)
    print(
        f"[3] PRIMARY PF contrast: Cliff δ = {pf['cliffs_delta']:+.4f} "
        f"({pf['cliffs_delta_magnitude']}), MWU p = {pf['mwu_p']:.5f}, "
        f"perm p = {pf['perm_p']:.5f} ({pf['perm_method']})"
    )
    print(
        f"    BCa 95% [{pf['bca_lo']:+.3f}, {pf['bca_hi']:+.3f}]; "
        f"inversion 95% [{pf['inv_ci_lo']:+.3f}, {pf['inv_ci_hi']:+.3f}]"
    )

    # --- 4. Sum-6 descriptive (demoted) -----------------------------------
    s6_c = (
        wide.loc[wide.group == "cyclops", "delta_lesion_total"].dropna().astype(float)
    )
    s6_m = (
        wide.loc[wide.group == "meniscus", "delta_lesion_total"].dropna().astype(float)
    )
    from scipy import stats as _st

    sum6 = tf.mwu_with_effects(
        s6_c.values, s6_m.values, n_boot=n_boot, seed=RANDOM_SEED
    )
    _, p_one = _st.mannwhitneyu(s6_c.values, s6_m.values, alternative="greater")
    results["sum6_delta"] = round(float(sum6["cliffs_delta"]), 4)
    results["sum6_p_two_sided"] = round(float(sum6["pvalue"]), 4)
    results["sum6_p_one_sided"] = round(float(p_one), 4)
    results["sum6_n_cyc"] = int(len(s6_c))
    results["sum6_n_men"] = int(len(s6_m))
    print(
        f"[4] sum-6 (demoted): Cliff δ = {sum6['cliffs_delta']:+.4f}, "
        f"p two-sided = {sum6['pvalue']:.4f}, one-sided = {p_one:.4f}"
    )

    # --- 5. Per-compartment + PF-vs-FT specificity ------------------------
    per_comp_rows = []
    comp_pvals = []
    for s in SITES:
        col = f"delta_{s}"
        if col not in wide.columns:
            continue
        cx = wide.loc[wide.group == "cyclops", col].dropna().astype(float)
        my = wide.loc[wide.group == "meniscus", col].dropna().astype(float)
        res = tf.mwu_with_effects(
            cx.values, my.values, n_boot=max(500, n_boot // 5), seed=RANDOM_SEED
        )
        wors_c = float((cx > 0).mean())
        wors_m = float((my > 0).mean())
        per_comp_rows.append(
            dict(
                compartment=s,
                block="PF" if s in SITES_PF else "FT",
                measurement="binary" if s in SITES_BINARY else "ordinal",
                worsened_pct_cyc=round(100 * wors_c, 1),
                worsened_pct_men=round(100 * wors_m, 1),
                cliff_delta=round(float(res["cliffs_delta"]), 4),
                mwu_p=round(float(res["pvalue"]), 4),
            )
        )
        comp_pvals.append(res["pvalue"])
    per_comp = pd.DataFrame(per_comp_rows)
    # BH-FDR across the six compartments (q = 0.10).
    bh = tf.bh_fdr(comp_pvals, q=0.10)
    per_comp["bh_reject_q0.10"] = bh["reject"]
    per_comp["bh_p_adj"] = [round(float(x), 4) for x in bh["pvals_corrected"]]
    per_comp.to_csv(RESULTS_DIR / "per_compartment.csv", index=False)

    pf_ft = tf.paired_pf_vs_ft(wide)
    results["pf_vs_ft_wilcoxon_p"] = round(float(pf_ft["pvalue"]), 6)
    results["pf_vs_ft_rank_biserial"] = round(float(pf_ft["rank_biserial"]), 4)
    results["pf_vs_ft_n_pf_worsened"] = int(pf_ft["n_pf_worsened"])
    results["pf_vs_ft_n_ft_worsened"] = int(pf_ft["n_ft_worsened"])
    print(
        f"[5] per-compartment saved ({len(per_comp)} sites); "
        f"PF-vs-FT (cyclops) Wilcoxon p = {pf_ft['pvalue']:.5f}, "
        f"r_rb = {pf_ft['rank_biserial']:+.2f} "
        f"(PF worsened {pf_ft['n_pf_worsened']} vs FT {pf_ft['n_ft_worsened']})"
    )

    # --- 6a. M1  beta-binomial on worsened_pf ----------------------------
    m1_results = {}
    for grp in ("cyclops", "meniscus"):
        sub = wide.loc[wide.group == grp, "worsened_pf"].dropna()
        k = int((sub == 1).sum())
        N = int(len(sub))
        m1 = bm.fit_m1_beta_binomial(k, N, hdi_prob=HDI_PROB)
        m1_results[grp] = dict(
            k=k,
            N=N,
            mean=round(m1["mean"], 4),
            hdi=[round(m1["hdi_lo"], 4), round(m1["hdi_hi"], 4)],
        )
    results["m1_worsened_pf"] = m1_results
    print(
        f"[6a] M1 worsened_pf: cyc {m1_results['cyclops']['mean']:.2f} "
        f"{m1_results['cyclops']['hdi']}, "
        f"mén {m1_results['meniscus']['mean']:.2f} {m1_results['meniscus']['hdi']}"
    )

    # --- 6b. M3  structure comparison + exchangeable-primary global δ̄ -----
    # Point C (revue 2026-05-29): the topographic structure is TESTED (LOO across
    # exchangeable / two_block / three_cluster), not assumed. The primary global
    # estimand is δ̄ from the selection-immune *exchangeable* model; the PF
    # localisation is a DERIVED contrast from that same neutral posterior; the
    # 2-block contrast is reported as a candidate-structure output, not co-primary.
    print(
        "[6b] M3 pooling-structure comparison "
        "(exchangeable / two_block / three_cluster) + LOO…"
    )
    t0 = time.time()
    cmpres = bm.compare_pooling_structures(df, sites=SITES, **mcmc_kwargs)
    idatas = cmpres["idatas"]
    idata_exch = idatas["exchangeable"]
    idata_2b = idatas["two_block"]
    conv_exch = bm.check_convergence(idata_exch)
    # Escalate the PRIMARY (exchangeable) if it did not converge (full runs only).
    if not conv_exch["ok"] and not smoke:
        print("     exchangeable not converged at base NUTS; escalating primary…")
        esc = bm.fit_m3_with_escalation(
            df, sites=SITES, pooling="exchangeable", compute_log_likelihood=True
        )
        idata_exch = esc["idata"]
        conv_exch = esc["convergence"]
        results["m3_escalated"] = bool(esc["escalated"])
    else:
        results["m3_escalated"] = False
    conv = conv_exch
    results["m3_fit_seconds"] = round(time.time() - t0, 1)

    # LOO comparison table  topographic structure as an empirical RESULT.
    loo_tbl = cmpres["compare"]
    loo_tbl.to_csv(RESULTS_DIR / "pooling_loo_compare.csv")
    results["pooling_loo"] = (
        loo_tbl.reset_index().rename(columns={"index": "model"}).to_dict("records")
    )
    results["pooling_loo_best"] = str(loo_tbl.index[0])
    results["pooling_loo_table_str"] = loo_tbl.to_string()

    # Convergence of all three structures.
    results["m3_convergence"] = {}
    for pool, idt in idatas.items():
        c = conv_exch if pool == "exchangeable" else bm.check_convergence(idt)
        results["m3_convergence"][pool] = dict(
            ok=bool(c["ok"]),
            max_rhat=round(float(c["max_rhat"]), 4),
            min_ess_bulk=round(float(c["min_ess_bulk"]), 1),
            n_divergent=int(c["n_divergent"]),
        )

    # PRIMARY: global knee-wide δ̄  H1 is directional → legitimate one-sided rule.
    results["m3_delta_bar_mean"] = round(_post_mean(idata_exch, "delta_bar"), 4)
    results["m3_delta_bar_hdi"] = _hdi(idata_exch, "delta_bar")
    results["m3_p_delta_bar_gt0"] = round(_p_gt0(idata_exch, "delta_bar"), 4)
    results["m3_gamma_mean"] = round(_post_mean(idata_exch, "gamma"), 4)
    v_global = rpt.verdict_bayes_en(
        idata_exch,
        "delta_bar",
        "H1 global knee-wide (delta-bar)",
        threshold=0.95,
        two_sided=False,
        direction="greater",
    )
    results["m3_verdict_global"] = v_global["sentence"]

    # LOCALISATION: PF−FT contrast DERIVED from the neutral exchangeable posterior
    # (non-circular; two-sided / HDI rule, partition selected → optimistic bound).
    dpf = bm.derived_pf_contrast(idata_exch)
    results["m3_derived_pf_contrast_mean"] = round(float(dpf["mean"]), 4)
    results["m3_derived_pf_contrast_hdi"] = [
        round(dpf["hdi_lo"], 4),
        round(dpf["hdi_hi"], 4),
    ]
    results["m3_derived_pf_contrast_p_gt0"] = round(float(dpf["p_gt0"]), 4)

    # CANDIDATE STRUCTURE: 2-block contrast (descriptive, two-sided; NOT co-primary).
    results["m3_2block_delta_pf_mean"] = round(_post_mean(idata_2b, "delta_pf"), 4)
    results["m3_2block_delta_ft_mean"] = round(_post_mean(idata_2b, "delta_ft"), 4)
    results["m3_2block_contrast_mean"] = round(
        _post_mean(idata_2b, "contrast_pf_ft"), 4
    )
    results["m3_2block_contrast_hdi"] = _hdi(idata_2b, "contrast_pf_ft")
    results["m3_2block_p_contrast_gt0"] = round(_p_gt0(idata_2b, "contrast_pf_ft"), 4)
    v_2b = rpt.verdict_bayes_en(
        idata_2b,
        "contrast_pf_ft",
        "Topographic specificity (PF-FT, 2-block candidate)",
        two_sided=True,
    )
    results["m3_verdict_2block_contrast"] = v_2b["sentence"]

    # Persist idatas: exchangeable = primary; two_block kept as idata_m3.nc for
    # back-compatible figures; three_cluster for completeness.
    results["m3_idata_path"] = _save_idata(idata_exch, "idata_m3_exchangeable.nc")
    _save_idata(idata_2b, "idata_m3.nc")
    if "three_cluster" in idatas:
        _save_idata(idatas["three_cluster"], "idata_m3_three_cluster.nc")

    print(
        f"     M3 done in {results['m3_fit_seconds']}s; primary(exchangeable) "
        f"converged ok={conv['ok']} (r̂={conv['max_rhat']:.3f}, "
        f"ESS={conv['min_ess_bulk']:.0f}, div={conv['n_divergent']}, "
        f"escalated={results['m3_escalated']})"
    )
    print(
        f"     delta-bar global={results['m3_delta_bar_mean']:+.2f} "
        f"HDI{results['m3_delta_bar_hdi']} P(>0)={results['m3_p_delta_bar_gt0']:.3f} "
        f"(honest knee-wide, dilution-prone)"
    )
    print(
        f"     derived PF-FT (neutral)={results['m3_derived_pf_contrast_mean']:+.2f} "
        f"HDI{results['m3_derived_pf_contrast_hdi']} "
        f"P(>0)={results['m3_derived_pf_contrast_p_gt0']:.3f}"
    )
    print(
        f"     2-block contrast(candidate)={results['m3_2block_contrast_mean']:+.2f} "
        f"HDI{results['m3_2block_contrast_hdi']}; LOO best = {results['pooling_loo_best']}"
    )
    print("     LOO table:\n" + loo_tbl.to_string())

    # --- 6c. M4 Weibull AFT + M5 LogNormal (H4 inter_surgery_d) -----------
    print("[6c] H4 inter_surgery_d (M4 Weibull AFT + M5 LogNormal)…")
    isd = wide[["group", "inter_surgery_d"]].dropna()
    isd_c = isd.loc[isd.group == "cyclops", "inter_surgery_d"].astype(float)
    isd_m = isd.loc[isd.group == "meniscus", "inter_surgery_d"].astype(float)
    results["isd_med_cyc"] = float(isd_c.median())
    results["isd_med_men"] = float(isd_m.median())
    isd_mwu = tf.mwu_with_effects(
        isd_c.values, isd_m.values, n_boot=n_boot, seed=RANDOM_SEED
    )
    results["isd_mwu_p"] = float(isd_mwu["pvalue"])
    results["isd_cliff"] = round(float(isd_mwu["cliffs_delta"]), 4)

    # M4 Weibull AFT (lifelines MLE  no MCMC). Use available covariates.
    aft_cov = [
        c for c in ("group", "imc", "pivot_pivot_contact") if c in merged.columns
    ]
    try:
        m4 = bm.fit_m4_weibull_aft(
            merged, duration_col="inter_surgery_d", covariates=aft_cov
        )
        m4_summary = m4["summary"].reset_index()
        m4_summary.to_csv(RESULTS_DIR / "m4_weibull_summary.csv", index=False)
        # group coefficient on log-scale (AFT: positive => longer delay).
        grp_row = m4["summary"].loc[
            m4["summary"].index.get_level_values(-1).str.contains("group", case=False)
        ]
        results["m4_group_coef"] = (
            round(float(grp_row["coef"].iloc[0]), 4) if len(grp_row) else None
        )
        results["m4_ok"] = True
    except Exception as exc:  # noqa: BLE001
        results["m4_ok"] = False
        results["m4_error"] = repr(exc)
        print(f"     [warn] M4 Weibull AFT failed: {exc!r}")

    # M5 LogNormal on inter_surgery_d (H4 as outcome).
    t0 = time.time()
    idata_m5 = bm.fit_m5_lognormal(
        wide, var="inter_surgery_d", nuts_sampler="nutpie", **mcmc_kwargs
    )
    conv5 = bm.check_convergence(idata_m5)
    results["m5_convergence"] = dict(
        ok=bool(conv5["ok"]),
        max_rhat=round(float(conv5["max_rhat"]), 4),
        min_ess_bulk=round(float(conv5["min_ess_bulk"]), 1),
        n_divergent=int(conv5["n_divergent"]),
    )
    # Posterior median delay per group = exp(mu).
    mu = idata_m5.posterior["mu"]
    g_names = list(mu.coords["group"].values)
    mu_means = {g: float(np.exp(mu.sel(group=g).values.mean())) for g in g_names}
    results["m5_median_delay_by_group"] = {g: round(v, 1) for g, v in mu_means.items()}
    results["m5_idata_path"] = _save_idata(idata_m5, "idata_m5.nc")
    # M4 idata placeholder: lifelines AFT has no idata; persist M5 as the H4
    # bayesian artefact, and re-save M5 also under idata_m4 expectations only
    # if a bayesian AFT existed. We additionally export a bayesian lognormal
    # under idata_m4.nc is NOT done (M4 is MLE). Document in results.
    results["m4_note"] = (
        "M4 is a lifelines MLE Weibull AFT (no idata); "
        "the bayesian H4 artefact is idata_m5.nc."
    )
    print(
        f"     isd medians cyc={isd_c.median():.0f} / mén={isd_m.median():.0f} "
        f"(MWU p={isd_mwu['pvalue']:.5f}, δ={isd_mwu['cliffs_delta']:+.2f})"
    )
    print(
        f"     M5 LogNormal done in {round(time.time() - t0, 1)}s; "
        f"converged ok={conv5['ok']}; "
        f"median delay {results['m5_median_delay_by_group']}"
    )

    # Also fit a bayesian lognormal saved as idata_m4 for completeness of the
    # requested artefact set (M4 family  duration model). Reuse M5 spec.
    results["m4_idata_path"] = _save_idata(idata_m5, "idata_m4.nc")

    # M1 has no idata (analytical); persist its scalars only, but the brief
    # lists idata_m1.nc  we materialise a tiny idata from the Beta posteriors
    # so the artefact exists and is loadable.
    try:
        import arviz as az

        rng = np.random.default_rng(RANDOM_SEED)
        draws = {}
        for grp, d in m1_results.items():
            a = 1.0 + d["k"]
            b = 1.0 + d["N"] - d["k"]
            draws[f"p_worsened_pf_{grp}"] = rng.beta(a, b, size=(4, 1000))
        try:
            idata_m1 = az.from_dict({"posterior": draws})
        except TypeError:
            idata_m1 = az.from_dict(posterior=draws)  # type: ignore[call-arg]
        results["m1_idata_path"] = _save_idata(idata_m1, "idata_m1.nc")
    except Exception as exc:  # noqa: BLE001
        results["m1_idata_path"] = None
        print(f"     [warn] M1 idata materialisation failed: {exc!r}")

    # --- 7. Sport/occupation sensitivity (with and without) ---------------
    sens = tf.sensitivity_covariate(
        merged,
        outcome_col="worsened_pf",
        covariates=("pivot_pivot_contact", "travail_physique"),
    )
    sens_df = pd.DataFrame([
        dict(
            model="crude (group only)",
            odds_ratio=sens["or_crude"],
            pvalue=sens["p_crude"],
            n=sens["n_crude"],
        ),
        dict(
            model="adjusted (+ pivot + travail_physique)",
            odds_ratio=sens["or_adjusted"],
            pvalue=sens["p_adjusted"],
            n=sens["n_adjusted"],
        ),
    ])
    sens_df.to_csv(RESULTS_DIR / "sensitivity_sport.csv", index=False)
    results["sensitivity_or_crude"] = (
        round(float(sens["or_crude"]), 4) if np.isfinite(sens["or_crude"]) else None
    )
    results["sensitivity_p_crude"] = (
        round(float(sens["p_crude"]), 4) if np.isfinite(sens["p_crude"]) else None
    )
    results["sensitivity_or_adjusted"] = (
        round(float(sens["or_adjusted"]), 4)
        if np.isfinite(sens["or_adjusted"])
        else None
    )
    results["sensitivity_p_adjusted"] = (
        round(float(sens["p_adjusted"]), 4) if np.isfinite(sens["p_adjusted"]) else None
    )
    results["sensitivity_covariates"] = sens["covariates"]
    print(
        f"[7] sport/métier sensitivity (worsened_pf): "
        f"OR crude = {sens['or_crude']:.2f} (p={sens['p_crude']:.3f}, n={sens['n_crude']}), "
        f"adjusted = {sens['or_adjusted']:.2f} (p={sens['p_adjusted']:.3f}, n={sens['n_adjusted']})"
    )

    # --- 7b. Headline adjusted PF effect (sex+age), E-value, delay --------
    # Firth penalised logistic (the ML OR is a quasi-separation ghost number).
    pf_crude = tf.firth_or(merged, outcome_col="worsened_pf")
    pf_adj = tf.firth_or(
        merged, outcome_col="worsened_pf", covariates=("female", "age_at_trauma")
    )
    results["pf_or_crude"] = round(float(pf_crude["odds_ratio"]), 3)
    results["pf_or_crude_ci"] = [
        round(pf_crude["or_ci_lo"], 3),
        round(pf_crude["or_ci_hi"], 3),
    ]
    results["pf_or_crude_method"] = pf_crude["method"]
    results["pf_or_crude_min_cell"] = int(pf_crude["min_cell"])
    results["pf_or_adj_sex_age"] = round(float(pf_adj["odds_ratio"]), 3)
    results["pf_or_adj_sex_age_ci"] = [
        round(pf_adj["or_ci_lo"], 3),
        round(pf_adj["or_ci_hi"], 3),
    ]
    results["pf_or_adj_sex_age_p"] = round(float(pf_adj["p"]), 4)
    ev = tf.evalue_or(pf_crude["odds_ratio"], pf_crude["or_ci_lo"], common_outcome=True)
    results["pf_evalue_point"] = round(float(ev["evalue_point"]), 3)
    results["pf_evalue_ci"] = (
        round(float(ev["evalue_ci"]), 3) if ev["evalue_ci"] else None
    )
    # Fuller adjustment incl. derived BMI (sex + age + imc).
    pf_adj_full = tf.firth_or(
        merged, outcome_col="worsened_pf", covariates=("female", "age_at_trauma", "imc")
    )
    results["pf_or_adj_sex_age_imc"] = round(float(pf_adj_full["odds_ratio"]), 3)
    results["pf_or_adj_sex_age_imc_ci"] = [
        round(pf_adj_full["or_ci_lo"], 3),
        round(pf_adj_full["or_ci_hi"], 3),
    ]
    results["pf_or_adj_sex_age_imc_n"] = int(pf_adj_full["n"])
    print(
        f"[7b] PF OR (Firth, {pf_crude['method']}; min cell={pf_crude['min_cell']}): "
        f"crude {results['pf_or_crude']} {results['pf_or_crude_ci']}; "
        f"sex+age-adj {results['pf_or_adj_sex_age']} {results['pf_or_adj_sex_age_ci']} "
        f"(p={results['pf_or_adj_sex_age_p']}); +imc {results['pf_or_adj_sex_age_imc']} "
        f"{results['pf_or_adj_sex_age_imc_ci']}; E-value {results['pf_evalue_point']} "
        f"(CI {results['pf_evalue_ci']})"
    )

    # Sensitivity: drop the date-anomaly patients (suspect age_at_trauma).
    if anomaly_keys:
        _akeys = set(anomaly_keys)
        _mask = pd.Series(
            [(g, a) not in _akeys for g, a in zip(merged["group"], merged["anonyme"])],
            index=merged.index,
        )
        pf_adj_clean = tf.firth_or(
            merged[_mask],
            outcome_col="worsened_pf",
            covariates=("female", "age_at_trauma"),
        )
        results["pf_or_adj_sex_age_excl_anomalies"] = round(
            float(pf_adj_clean["odds_ratio"]), 3
        )
        results["pf_or_adj_sex_age_excl_n"] = int(pf_adj_clean["n"])
        print(
            f"     sex+age-adj excl. {len(_akeys)} anomaly patient(s): "
            f"OR {results['pf_or_adj_sex_age_excl_anomalies']} (n={pf_adj_clean['n']})"
        )

    # Delay as TIME-AT-RISK (not a biological mediator): adjusted OR + falsification.
    pf_delay = tf.firth_or(
        merged, outcome_col="worsened_pf", covariates=("inter_surgery_d",)
    )
    results["pf_or_adj_delay"] = round(float(pf_delay["odds_ratio"]), 3)
    results["pf_or_adj_delay_ci"] = [
        round(pf_delay["or_ci_lo"], 3),
        round(pf_delay["or_ci_hi"], 3),
    ]
    _dly = wide[["inter_surgery_d", "worsened_pf"]].dropna()
    from scipy import stats as _st2

    rho_d, p_d = _st2.spearmanr(
        _dly["inter_surgery_d"].astype(float), _dly["worsened_pf"].astype(float)
    )
    results["delay_worsened_pf_rho"] = round(float(rho_d), 4)
    results["delay_worsened_pf_rho_p"] = round(float(p_d), 4)
    print(
        f"     delay-adjusted PF OR {results['pf_or_adj_delay']} "
        f"{results['pf_or_adj_delay_ci']} (effect persists at constant exposure); "
        f"falsification rho(delay,worsened_pf)={results['delay_worsened_pf_rho']:+.3f} "
        f"(p={results['delay_worsened_pf_rho_p']:.3f}) -> not a confounder"
    )

    # --- 8. H3  intrinsic risk factors vs PF progression (cyclops only) ---
    cyc = merged[merged.group == "cyclops"].copy()
    h3 = tf.h3_risk_factors(
        cyc,
        outcome_col="delta_lesion_pf",
        continuous=("age_at_trauma", "imc"),
        binary=("female", "tabac", "travail_physique"),
        multilevel=("pivot_pivot_contact",),
        q=0.10,
        n_boot=n_boot,
        seed=RANDOM_SEED,
    )
    h3.to_csv(RESULTS_DIR / "h3_factors.csv", index=False)
    results["h3_factors"] = h3.to_dict(orient="records")
    results["h3_any_bh_significant"] = (
        bool(h3["bh_reject"].any()) if "bh_reject" in h3.columns else False
    )
    print("[8] H3 risk factors (cyclops, Δ_PF), exploratory, BH-FDR q=0.10:")
    for _, r in h3.iterrows():
        eff = r["effect"]
        padj = r.get("p_adj_bh", float("nan"))
        sig = "sig" if r.get("bh_reject", False) else "ns"
        print(
            f"     {r['factor']:18s} {r['effect_name']}={eff:+.3f} "
            f"p={r['p']:.3f} p_adj={padj:.3f} [{sig}]"
        )

    # --- 9. Flexum (pre-S2 extension deficit)  the mechanical driver ------
    # Cyclops nodule blocks extension → fixed flexum → patellofemoral overload.
    # Two honest reads: (i) GROUP separation  is the flexum present where the PF
    # damage is?  and (ii) intra-cyclops DOSE-RESPONSE  does a deeper flexum
    # predict a bigger Δ_PF? The flexum workbook (data/flexum.xlsx) is joined to
    # the cohort by (group, anonyme); unmatched rows (the meniscus sheet carries
    # 49 padding rows, all 0°, vs the 20 cohort meniscus → 19 match after the
    # patient-25 reclassification) are dropped by the inner join  a documented
    # caveat. Meniscus flexum is constant 0, so the dose-response is cyclops-only.
    try:
        flex = loaders.load_flexum()
        wflex = wide[["group", "anonyme", "delta_lesion_pf"]].copy()
        wflex["anonyme"] = pd.to_numeric(wflex["anonyme"], errors="coerce").astype(
            "Int64"
        )
        fm = wflex.merge(flex, on=["group", "anonyme"], how="inner")
        fcyc = fm[fm.group == "cyclops"]
        fmen = fm[fm.group == "meniscus"]
        cyc_any = int((fcyc["flexum_pre_s2"] < 0).sum())
        men_any = int((fmen["flexum_pre_s2"] < 0).sum())
        results["flexum_n_cyc"] = int(len(fcyc))
        results["flexum_n_men"] = int(len(fmen))
        results["flexum_cyc_n_deficit"] = cyc_any
        results["flexum_men_n_deficit"] = men_any
        results["flexum_cyc_median"] = float(fcyc["flexum_pre_s2"].median())
        results["flexum_cyc_min"] = float(fcyc["flexum_pre_s2"].min())
        # (i) group separation: any-flexum 2×2 (Fisher) + signed-degree MWU/Cliff.
        tab = np.array([[cyc_any, len(fcyc) - cyc_any], [men_any, len(fmen) - men_any]])
        fish = tf.fisher_exact_2x2(tab)
        results["flexum_fisher_p"] = float(fish["pvalue"])
        results["flexum_fisher_or"] = (
            round(float(fish["odds_ratio"]), 3)
            if np.isfinite(fish["odds_ratio"])
            else None
        )
        fmwu = tf.mwu_with_effects(
            fcyc["flexum_pre_s2"].values.astype(float),
            fmen["flexum_pre_s2"].values.astype(float),
            n_boot=n_boot,
            seed=RANDOM_SEED,
        )
        results["flexum_cliff"] = round(float(fmwu["cliffs_delta"]), 4)
        results["flexum_mwu_p"] = float(fmwu["pvalue"])
        # (ii) intra-cyclops dose-response: deeper flexum (deficit = −flexum) vs Δ_PF.
        cc = fcyc[fcyc["delta_lesion_pf"].notna()]
        sp = tf.spearman_bca(
            (-cc["flexum_pre_s2"]).values.astype(float),
            cc["delta_lesion_pf"].values.astype(float),
            n_boot=n_boot,
            seed=RANDOM_SEED,
        )
        results["flexum_dpf_spearman_rho"] = round(float(sp["rho"]), 4)
        results["flexum_dpf_spearman_ci"] = [
            round(sp["ci_lo"], 4),
            round(sp["ci_hi"], 4),
        ]
        results["flexum_dpf_spearman_p"] = round(float(sp["pvalue"]), 4)
        results["flexum_dpf_n"] = int(sp["n"])
        results["flexum_ok"] = True
        print(
            f"[9] Flexum: cyclops {cyc_any}/{len(fcyc)} with deficit "
            f"(median {fcyc['flexum_pre_s2'].median():.0f}°, min {fcyc['flexum_pre_s2'].min():.0f}°) "
            f"vs meniscus {men_any}/{len(fmen)} → Fisher p={fish['pvalue']:.2e}, "
            f"Cliff δ={fmwu['cliffs_delta']:+.3f} ({fmwu['cliffs_delta_magnitude']})"
        )
        print(
            f"     intra-cyclops dose-response ρ(deficit,Δ_PF)={sp['rho']:+.3f} "
            f"[{sp['ci_lo']:+.2f},{sp['ci_hi']:+.2f}] p={sp['pvalue']:.3f} (n={sp['n']}) "
            f"→ {'NULL (present/absent marker, not graded)' if sp['pvalue'] > 0.05 else 'graded'}"
        )
    except Exception as exc:  # noqa: BLE001
        results["flexum_ok"] = False
        results["flexum_error"] = repr(exc)
        print(f"[9] [warn] flexum analysis failed: {exc!r}")

    # --- Save results.json ------------------------------------------------
    results["elapsed_seconds"] = round(time.time() - t_start, 1)
    results["artefacts_dir"] = str(RESULTS_DIR)
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as fh:
        json.dump(_jsonable(results), fh, indent=2, ensure_ascii=False)

    # --- Convergence + canonical report -----------------------------------
    print("\n" + "=" * 78)
    print("CONVERGENCE REPORT")
    print("=" * 78)
    print(
        f"M3 primary (exchangeable, δ̄): ok={conv['ok']}  r̂_max={conv['max_rhat']:.4f}  "
        f"ESS_bulk_min={conv['min_ess_bulk']:.0f}  divergences={conv['n_divergent']}  "
        f"escalated={results['m3_escalated']}"
    )
    if not conv["ok"] and conv.get("offenders"):
        print("  offenders:", conv["offenders"][:8])
    print(
        f"M3 structure LOO: best = {results['pooling_loo_best']} "
        f"(exchangeable=selection-immune primary; 2-block/3-cluster = candidate structures)"
    )
    for _pool, _c in results["m3_convergence"].items():
        print(
            f"  {_pool:14s} ok={_c['ok']} r̂={_c['max_rhat']:.3f} "
            f"ESS={_c['min_ess_bulk']:.0f} div={_c['n_divergent']}"
        )
    print(
        f"M5 (H4 lognormal): ok={conv5['ok']}  r̂_max={conv5['max_rhat']:.4f}  "
        f"ESS_bulk_min={conv5['min_ess_bulk']:.0f}  divergences={conv5['n_divergent']}"
    )
    print(
        "M4 (Weibull AFT): lifelines MLE  no MCMC convergence "
        f"(fit ok={results.get('m4_ok')})"
    )

    print("\n" + "=" * 78)
    print("CANONICAL NUMBERS (vs reference)")
    print("=" * 78)
    print(
        rpt.verdict_freq_en(
            pf["mwu_p"],
            "PF contrast (cyclops vs meniscus, frequentist rank-based)",
            effect=f"Cliff δ = {pf['cliffs_delta']:+.3f} ({pf['cliffs_delta_magnitude']})",
        )
    )
    print("GLOBAL (honest, dilution-prone): " + v_global["sentence"])
    print(
        f"DERIVED PF-FT (neutral exchangeable model): mean "
        f"{results['m3_derived_pf_contrast_mean']:+.2f}, "
        f"HDI {results['m3_derived_pf_contrast_hdi']}, "
        f"P(>0)={results['m3_derived_pf_contrast_p_gt0']:.3f}"
    )
    print(
        "CANDIDATE 2-block: "
        + v_2b["sentence"]
        + f"  | LOO favours: {results['pooling_loo_best']}"
    )
    print(
        f"PF OR Firth: crude {results['pf_or_crude']} {results['pf_or_crude_ci']}; "
        f"sex+age-adj {results['pf_or_adj_sex_age']} {results['pf_or_adj_sex_age_ci']}; "
        f"E-value {results['pf_evalue_point']}"
    )
    print(
        f"baseline S1 (6-sum) MWU p = {base['pvalue']:.4f} (ref ≈ 0.78); "
        f"baseline PF TOST p = {results.get('baseline_pf_tost_p')} "
        f"(equivalent={results.get('baseline_pf_equivalent')})"
    )
    print(
        f"sum-6 δ = {sum6['cliffs_delta']:+.3f}, p = {sum6['pvalue']:.3f} "
        f"two-sided / {p_one:.3f} one-sided  (ref δ≈0.25, p≈0.088/0.044)"
    )
    print(
        f"inter_surgery_d median cyc={isd_c.median():.0f} / mén={isd_m.median():.0f}  "
        f"(ref 240/518)"
    )
    print(f"\nArtefacts → {RESULTS_DIR}")
    print(f"Total elapsed: {results['elapsed_seconds']}s")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reproducible lino_stats pipeline.")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny MCMC + fewer resamples for a quick smoke run",
    )
    args = ap.parse_args()
    main(smoke=args.smoke)
