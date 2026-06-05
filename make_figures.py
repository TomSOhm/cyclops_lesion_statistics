#!/usr/bin/env python
"""make_figures.py — regenerate every manuscript figure (300 dpi, English).

Reproducible (``RANDOM_SEED = 42``), idempotent figure builder for the
cyclops-vs-meniscus knee-surgery study. It reads the canonical analysis
artefacts — ``results/results.json`` (all scalars), the per-compartment /
Table-1 CSVs, the posterior NetCDFs (``idata_m3.nc``, ``idata_m5.nc``) — and
re-derives the patient-level frames straight from the preprocessing pipeline,
then writes the figure manifest to ``figures/``:

    fig1_baseline_balance.png      Love plot of baseline SMDs + S1 equivalence
    fig2_pf_progression.png        PRIMARY: patellofemoral Δ raincloud
    fig3_per_compartment.png       worsening % per compartment (PF vs FT blocks)
    fig4_topographic_specificity.png  paired ΔPF vs ΔFT within cyclops
    fig5_m3_forest.png             M3 posteriors (δ_PF, δ_FT, contrast, γ), HDI 94%
    fig6_m3_diagnostics.png        M3 trace + rank + convergence banner
    fig7_h4_delay.png              inter-surgical delay ECDF + LogNormal fit
    figS1_slopegraph_pf.png        (supplementary) PF score S1→S2 by group

Every number is sourced from the artefacts — nothing is hard-coded. Re-running
overwrites the PNGs in place.

It edits/creates nothing outside ``figures/`` (chart-expert scope) and never
touches ``run_all.py`` or the analysis modules other than importing them.

Run::

    C:/Users/salem/miniconda3/envs/dev/python.exe make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# UTF-8 console (δ / accented compartment names) on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

# Headless backend (no display in the run environment).
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from constants import RANDOM_SEED, SITES_PF, SITES_FT  # noqa: E402
import loaders  # noqa: E402
import preprocessing as pp  # noqa: E402
import tests_freq as tf  # noqa: E402
import viz  # noqa: E402

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_results() -> dict:
    with open(RESULTS_DIR / "results.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_idata(name: str):
    """Load a NetCDF posterior, returning ``None`` if absent/unreadable."""
    import arviz as az

    path = RESULTS_DIR / name
    if not path.exists():
        print(f"  [warn] {name} not found — skipping dependent figure.")
        return None
    try:
        return az.from_netcdf(str(path))
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not read {name}: {exc!r}")
        return None


def _build_frames():
    """Re-derive the wide / patient frames via the canonical pipeline."""
    df = loaders.load_combined()
    df = pp.apply_date_hygiene(df)      # composite-key bug fix (consensus I.1)
    df = pp.add_derived(df)
    wide = pp.to_wide(df)
    patient = pp.to_patient(df)
    return df, wide, patient


def _smd_binary(a: pd.Series, b: pd.Series) -> float:
    """Binary standardised mean difference (pooled-prevalence denominator)."""
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pa, pb = float(a.mean()), float(b.mean())
    sp = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2.0)
    return float((pa - pb) / sp) if sp > 0 else float("nan")


# ---------------------------------------------------------------------------
# Figure builders (each returns the written Path)
# ---------------------------------------------------------------------------

def make_fig0a(patient: pd.DataFrame) -> Path:
    """First-glance demographic panel by group (age, % female, BMI)."""
    fig = viz.descriptive_demographics(patient)
    out = viz.save_pub_fig(fig, "fig0a_demographics", FIG_DIR)
    plt.close(fig)
    return out


def make_fig0b(wide: pd.DataFrame) -> Path:
    """First-glance baseline cartilage view by group at S1."""
    fig = viz.descriptive_lesion_baseline(wide)
    out = viz.save_pub_fig(fig, "fig0b_baseline_lesions", FIG_DIR)
    plt.close(fig)
    return out


def _covariate_smd_rows(patient: pd.DataFrame) -> list[tuple[str, float]]:
    """Patient-covariate SMD rows (Cyclops − Meniscus) shared by fig1 / fig1c."""
    c = patient[patient.group == "cyclops"]
    m = patient[patient.group == "meniscus"]
    rows = [
        ("Female (sex)", _smd_binary(c["female"], m["female"])),
        ("Age at trauma", tf.smd_continuous(
            pd.to_numeric(c["age_at_trauma"], errors="coerce").values,
            pd.to_numeric(m["age_at_trauma"], errors="coerce").values)),
        ("Pivot / contact sport", _smd_binary(
            (pd.to_numeric(c["pivot_pivot_contact"], errors="coerce") >= 1).astype(float),
            (pd.to_numeric(m["pivot_pivot_contact"], errors="coerce") >= 1).astype(float))),
        ("Physical occupation", _smd_binary(c["travail_physique"], m["travail_physique"])),
        ("Smoking", _smd_binary(c["tabac"], m["tabac"])),
        ("BMI", tf.smd_continuous(
            pd.to_numeric(c["imc"], errors="coerce").values,
            pd.to_numeric(m["imc"], errors="coerce").values)),
    ]
    return [(lbl, v) for lbl, v in rows if np.isfinite(v)]


def make_fig1(patient: pd.DataFrame, results: dict) -> Path:
    """View 1 — Love plot of baseline COVARIATE SMDs (Cyclops − Meniscus)."""
    rows = _covariate_smd_rows(patient)
    fig = viz.love_plot_smd(
        rows, baseline_score_p=float(results.get("baseline_s1_p", float("nan"))),
        title="Baseline covariate balance (SMD) — view 1: covariables",
    )
    out = viz.save_pub_fig(fig, "fig1_baseline_balance", FIG_DIR)
    plt.close(fig)
    return out


def make_fig1b(results: dict) -> Path | None:
    """View 2 — Baseline CARTILAGE balance, PF + FT only, SAME love-plot style."""
    rows = []
    for lvl, lbl in (("pf", "PF cartilage S1 (trochlea+patella)"),
                     ("ft", "FT cartilage S1 (tibial+condylar)")):
        smd = results.get(f"baseline_{lvl}_smd")
        if smd is None:
            continue
        rows.append((lbl, float(smd)))
    if not rows:
        return None
    fig = viz.love_plot_smd(
        rows, title="Baseline cartilage balance (SMD) — view 2: PF / FT blocks",
    )
    out = viz.save_pub_fig(fig, "fig1b_baseline_cartilage", FIG_DIR)
    plt.close(fig)
    return out


def make_fig1c(patient: pd.DataFrame, results: dict) -> Path:
    """View 3 — All covariables + GLOBAL cartilage, SAME love-plot style."""
    rows = _covariate_smd_rows(patient)
    smd_global = results.get("baseline_total_smd")
    if smd_global is not None:
        rows.append(("Cartilage S1 (global 6-sum)", float(smd_global)))
    fig = viz.love_plot_smd(
        rows, title="Baseline balance (SMD) — view 3: all covariables + global cartilage",
    )
    out = viz.save_pub_fig(fig, "fig1c_baseline_global", FIG_DIR)
    plt.close(fig)
    return out


def make_fig2(wide: pd.DataFrame, results: dict) -> Path:
    """PRIMARY patellofemoral progression raincloud."""
    m1 = results.get("m1_worsened_pf", {})
    worsened = {
        "cyclops": 100.0 * m1.get("cyclops", {}).get("mean", float("nan")),
        "meniscus": 100.0 * m1.get("meniscus", {}).get("mean", float("nan")),
    }
    # Prefer the exact observed proportions (k/N) when available.
    for g in ("cyclops", "meniscus"):
        d = m1.get(g)
        if d and d.get("N"):
            worsened[g] = 100.0 * d["k"] / d["N"]
    fig = viz.raincloud_progression(
        wide, value_col="delta_lesion_pf",
        worsened_pct=worsened,
        cliff_delta=results.get("pf_cliff_delta"),
        perm_p=results.get("pf_perm_p"),
        title="Patellofemoral progression (primary)",
        ylabel="Δ patellofemoral lesion score (S2 − S1)",
    )
    out = viz.save_pub_fig(fig, "fig2_pf_progression", FIG_DIR)
    plt.close(fig)
    return out


def make_fig3() -> Path:
    """Per-compartment worsening % (PF vs FT blocks)."""
    per_comp = pd.read_csv(RESULTS_DIR / "per_compartment.csv")
    fig = viz.per_compartment_bars(per_comp)
    out = viz.save_pub_fig(fig, "fig3_per_compartment", FIG_DIR)
    plt.close(fig)
    return out


def make_fig4(wide: pd.DataFrame, results: dict) -> Path:
    """Topographic specificity within cyclops (paired ΔPF vs ΔFT)."""
    fig = viz.topographic_specificity(
        wide,
        n_pf_worsened=results.get("pf_vs_ft_n_pf_worsened"),
        n_ft_worsened=results.get("pf_vs_ft_n_ft_worsened"),
        rank_biserial=results.get("pf_vs_ft_rank_biserial"),
        wilcoxon_p=results.get("pf_vs_ft_wilcoxon_p"),
    )
    out = viz.save_pub_fig(fig, "fig4_topographic_specificity", FIG_DIR)
    plt.close(fig)
    return out


def make_fig5(idata_m3) -> Path | None:
    """M3 posterior forest (δ_PF, δ_FT, contrast, γ)."""
    if idata_m3 is None:
        return None
    fig = viz.forest_m3(idata_m3)
    out = viz.save_pub_fig(fig, "fig5_m3_forest", FIG_DIR)
    plt.close(fig)
    return out


def make_fig6(idata_m3, results: dict) -> Path | None:
    """M3 convergence diagnostics (trace + rank + banner)."""
    if idata_m3 is None:
        return None
    fig = viz.diagnostics_m3(
        idata_m3, convergence=results.get("m3_convergence"),
    )
    out = viz.save_pub_fig(fig, "fig6_m3_diagnostics", FIG_DIR)
    plt.close(fig)
    return out


def make_fig7(wide: pd.DataFrame, idata_m5, results: dict) -> Path:
    """Inter-surgical delay ECDF + LogNormal fit (H4)."""
    medians = {
        "cyclops": results.get("isd_med_cyc"),
        "meniscus": results.get("isd_med_men"),
    }
    medians = {g: v for g, v in medians.items() if v is not None}
    fig = viz.delay_ecdf_fit(
        wide, idata=idata_m5, medians=medians or None,
    )
    out = viz.save_pub_fig(fig, "fig7_h4_delay", FIG_DIR)
    plt.close(fig)
    return out


def make_figS1(wide: pd.DataFrame) -> Path:
    """Supplementary PF slopegraph S1 → S2 by group."""
    fig = viz.slopegraph_pf_mpl(wide)
    out = viz.save_pub_fig(fig, "figS1_slopegraph_pf", FIG_DIR)
    plt.close(fig)
    return out


def make_fig10(wide: pd.DataFrame, results: dict) -> Path | None:
    """Flexum panel: group separation + intra-cyclops dose-response."""
    if not results.get("flexum_ok"):
        return None
    flex = loaders.load_flexum()
    wj = wide[["group", "anonyme", "delta_lesion_pf"]].copy()
    wj["anonyme"] = pd.to_numeric(wj["anonyme"], errors="coerce").astype("Int64")
    fm = wj.merge(flex, on=["group", "anonyme"], how="inner")
    fig = viz.flexum_panel(
        fm,
        spearman_rho=results.get("flexum_dpf_spearman_rho"),
        spearman_ci=results.get("flexum_dpf_spearman_ci"),
        spearman_p=results.get("flexum_dpf_spearman_p"),
        fisher_p=results.get("flexum_fisher_p"),
    )
    out = viz.save_pub_fig(fig, "fig10_flexum", FIG_DIR)
    plt.close(fig)
    return out


def _hdi94(arr) -> tuple[float, float]:
    a = np.sort(np.asarray(arr).ravel())
    n = len(a)
    k = max(1, int(np.floor(0.94 * n)))
    w = a[k - 1:] - a[: n - k + 1]
    i = int(np.argmin(w))
    return float(a[i]), float(a[i + k - 1])


def make_fig8(idata_exch, results: dict) -> Path | None:
    """Honest global δ̄ (diluted) vs localised PF — the de-biasing forest.

    From the **neutral exchangeable** M3: the knee-wide δ̄ (includes 0 → honest,
    not overclaimed), the six per-compartment δ_c, and the DERIVED PF−FT contrast
    (non-circular). Shows the localisation emerging without any imposed partition.
    """
    if idata_exch is None:
        return None
    post = idata_exch.posterior
    rows = []
    m = float(post["delta_bar"].values.mean())
    lo, hi = _hdi94(post["delta_bar"].values)
    p = float((post["delta_bar"].values > 0).mean())
    rows.append(("delta-bar (global, knee-wide)", m, lo, hi, p, "global"))
    dc = post["delta_comp"]
    comp_dim = [d for d in dc.dims if d not in ("chain", "draw")][0]
    for c in [str(x) for x in dc[comp_dim].values]:
        vals = dc.sel({comp_dim: c}).values
        lo_, hi_ = _hdi94(vals)
        blk = "PF" if c in SITES_PF else "FT"
        rows.append((f"delta {c}", float(vals.mean()), lo_, hi_,
                     float((vals > 0).mean()), blk))
    m = results["m3_derived_pf_contrast_mean"]
    lo, hi = results["m3_derived_pf_contrast_hdi"]
    rows.append(("PF - FT (derived)", m, lo, hi,
                 results["m3_derived_pf_contrast_p_gt0"], "contrast"))

    colors = {"global": "#444444", "PF": "#1b7837", "FT": "#999999", "contrast": "#762a83"}
    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lbl, mm, lo_, hi_, pp_, blk) in zip(y, rows):
        ax.plot([lo_, hi_], [yi, yi], color=colors[blk], lw=2.2, solid_capstyle="round")
        ax.plot(mm, yi, "o", color=colors[blk], ms=6)
        ax.text(hi_, yi, f"  P(>0)={pp_:.2f}", va="center", fontsize=8, color=colors[blk])
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Group x Time interaction δ (logit scale), 94% HDI")
    ax.set_title("M3: honest global δ̄ (diluted, includes 0) vs localised PF\n"
                 "(derived from the neutral exchangeable model — no PF/FT imposed)",
                 fontsize=10)
    fig.tight_layout()
    out = viz.save_pub_fig(fig, "fig8_global_vs_localised", FIG_DIR)
    plt.close(fig)
    return out


def make_fig9(results: dict) -> Path | None:
    """Firth odds-ratio forest for PF worsening (crude → adjusted → delay).

    Replaces the quasi-separation ML ghost (OR=24, meniscus 1/19) with stable
    Firth estimates and shows the effect survives sex/age/BMI adjustment; the
    E-value quantifies robustness to unmeasured confounding.
    """
    items = [
        ("Crude", results.get("pf_or_crude"), results.get("pf_or_crude_ci")),
        ("Adj. sex + age", results.get("pf_or_adj_sex_age"), results.get("pf_or_adj_sex_age_ci")),
        ("Adj. sex + age + BMI", results.get("pf_or_adj_sex_age_imc"), results.get("pf_or_adj_sex_age_imc_ci")),
        ("Adj. delay (time-at-risk)", results.get("pf_or_adj_delay"), results.get("pf_or_adj_delay_ci")),
    ]
    items = [(l, o, ci) for l, o, ci in items if o and ci]
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 0.6 * len(items) + 1.8))
    y = np.arange(len(items))[::-1]
    for yi, (lbl, o, ci) in zip(y, items):
        ax.plot(ci, [yi, yi], color="#1b7837", lw=2.2, solid_capstyle="round")
        ax.plot(o, yi, "o", color="#1b7837", ms=6)
        ax.text(ci[1], yi, f"  {o:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]", va="center", fontsize=8)
    ax.axvline(1, color="k", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items])
    ax.set_xlabel("Odds ratio for patellofemoral worsening (Firth penalised, log scale)")
    ev, evci = results.get("pf_evalue_point"), results.get("pf_evalue_ci")
    ax.set_title(f"PF-worsening OR — Firth (E-value {ev} / CI-bound {evci}; "
                 "ML OR=24 was quasi-separation)", fontsize=10)
    fig.tight_layout()
    out = viz.save_pub_fig(fig, "fig9_firth_or_forest", FIG_DIR)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> list[Path]:
    np.random.seed(RANDOM_SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    viz.set_pub_style()

    print("=" * 72)
    print(f"make_figures.py — seed={RANDOM_SEED}, dpi={viz.PUB_DPI}")
    print("=" * 72)

    results = _load_results()
    _, wide, patient = _build_frames()
    idata_m3 = _load_idata("idata_m3.nc")                  # two_block (back-compat)
    idata_exch = _load_idata("idata_m3_exchangeable.nc")   # neutral primary (δ̄)
    idata_m5 = _load_idata("idata_m5.nc")

    written: list[Path] = []
    builders = [
        ("fig0a_demographics", lambda: make_fig0a(patient)),
        ("fig0b_baseline_lesions", lambda: make_fig0b(wide)),
        ("fig1_baseline_balance", lambda: make_fig1(patient, results)),
        ("fig1b_baseline_cartilage", lambda: make_fig1b(results)),
        ("fig1c_baseline_global", lambda: make_fig1c(patient, results)),
        ("fig2_pf_progression", lambda: make_fig2(wide, results)),
        ("fig3_per_compartment", make_fig3),
        ("fig4_topographic_specificity", lambda: make_fig4(wide, results)),
        ("fig5_m3_forest", lambda: make_fig5(idata_m3)),
        ("fig6_m3_diagnostics", lambda: make_fig6(idata_m3, results)),
        ("fig7_h4_delay", lambda: make_fig7(wide, idata_m5, results)),
        ("fig8_global_vs_localised", lambda: make_fig8(idata_exch, results)),
        ("fig9_firth_or_forest", lambda: make_fig9(results)),
        ("fig10_flexum", lambda: make_fig10(wide, results)),
        ("figS1_slopegraph_pf", lambda: make_figS1(wide)),
    ]
    for name, fn in builders:
        try:
            path = fn()
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"  [FAIL] {name}: {exc!r}")
            continue
        if path is None:
            print(f"  [skip] {name} (missing dependency)")
            continue
        size = path.stat().st_size if path.exists() else 0
        status = "OK " if size > 0 else "EMPTY"
        print(f"  [{status}] {path.name:32s} {size/1024:7.1f} KB")
        written.append(path)

    print("-" * 72)
    print(f"Wrote {len(written)} figures to {FIG_DIR}")
    for p in written:
        print(f"  {p}")
    return written


if __name__ == "__main__":
    main()
