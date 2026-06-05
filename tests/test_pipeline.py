"""Sanity asserts on the data pipeline.

Run with::

    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from constants import (
    BLOCKS,
    GROUPS,
    N_CYCLOPS,
    N_CYCLOPS_ANALYSABLE,
    N_CYCLOPS_ROWS,
    N_MENISCUS,
    N_MENISCUS_ROWS,
    N_ROWS,
    N_TOTAL,
    N_TOTAL_ANALYSABLE,
    RANDOM_SEED,
    SCORE_MAX,
    SCORE_MAX_COLLAPSED,
    SCORE_MIN,
    SITES,
    SITES_BINARY,
    SITES_FT,
    SITES_ORDINAL,
    SITES_PF,
)
import loaders
import preprocessing as pp
import reporting as rpt
import tests_freq as tf
from loaders import _normalise_cols


# --- Loaders ----------------------------------------------------------------

def test_load_meniscus_shape():
    df = loaders.load_meniscus()
    assert df.shape[0] == N_MENISCUS_ROWS == 40
    assert df["anonyme"].nunique() == N_MENISCUS == 20
    assert (df["group"] == "meniscus").all()


def test_load_cyclops_shape():
    df = loaders.load_cyclops()
    assert df.shape[0] == N_CYCLOPS_ROWS == 98
    assert df["anonyme"].nunique() == N_CYCLOPS == 49
    assert (df["group"] == "cyclops").all()


def test_load_combined_shape():
    df = loaders.load_combined()
    assert df.shape[0] == N_ROWS == 138
    # group counts
    g = df["group"].value_counts()
    assert g["meniscus"] == N_MENISCUS_ROWS
    assert g["cyclops"] == N_CYCLOPS_ROWS


def test_load_flexum_shape_and_separation():
    """Flexum workbook loads long; cyclops carry a deficit, meniscus do not."""
    fx = loaders.load_flexum()
    assert {"anonyme", "group", "flexum_pre_s2"} <= set(fx.columns)
    cyc = fx[fx["group"] == "cyclops"]["flexum_pre_s2"]
    men = fx[fx["group"] == "meniscus"]["flexum_pre_s2"]
    # Cyclops have ≤0 values with some real deficit; meniscus are all 0 (no nodule).
    assert (cyc <= 0).all() and int((cyc < 0).sum()) == 29
    assert (men == 0).all()
    assert cyc.min() == -10


def test_normalised_columns():
    df = loaders.load_combined()
    expected = {
        "anonyme", "sexe", "date_de_naissance", "taille", "poids", "imc",
        "pivot_pivot_contact", "travail_physique", "tabac",
        "date_du_trauma", "date_chir", "group",
    } | set(SITES)
    missing = expected - set(df.columns)
    assert not missing, f"Missing cols: {missing}"


# --- Preprocessing ----------------------------------------------------------

def test_surgery_num_balanced():
    df = loaders.load_combined()
    df = pp.add_surgery_num(df)
    counts = df["surgery_num"].value_counts().sort_index()
    assert int(counts.loc[1]) == N_TOTAL == 69
    assert int(counts.loc[2]) == N_TOTAL == 69


def test_lesion_total_range():
    df = loaders.load_combined()
    df = pp.add_derived(df)
    valid = df["lesion_total"].dropna()
    # 6 sites × SCORE_MAX upper bound
    assert valid.between(0, 6 * SCORE_MAX).all(), (
        f"lesion_total outside [0, {6 * SCORE_MAX}]: "
        f"min={int(valid.min())}, max={int(valid.max())}"
    )


def test_composite_patient_key_balanced():
    """Each physical patient = (group, anonyme) appears exactly twice (S1, S2).

    Sentinel for consensus point I.1: the ``Anonyme`` id is reused across the
    two sheets (19 shared ids); grouping by the composite key must yield 69
    patients each with 2 rows. A failure here means a sheet-crossing merge.
    """
    df = loaders.load_combined()
    sizes = df.groupby(["group", "anonyme"]).size()
    assert (sizes == 2).all(), (
        f"Some (group, anonyme) keys do not have exactly 2 rows: "
        f"{sizes[sizes != 2].to_dict()}"
    )
    assert len(sizes) == N_TOTAL == 69
    # 19 ids are shared between the two sheets (the bug's root cause).
    m_ids = set(df.loc[df.group == "meniscus", "anonyme"].unique())
    c_ids = set(df.loc[df.group == "cyclops", "anonyme"].unique())
    assert len(m_ids & c_ids) == 19


def test_date_hygiene_uses_composite_key():
    """apply_date_hygiene must not collapse dates across the two sheets.

    A shared id (e.g. 1) has different birth dates in meniscus vs cyclops;
    after hygiene those must remain distinct (grouped by composite key).
    """
    df = loaders.load_combined()
    hy = pp.apply_date_hygiene(df)
    # Pick a shared id and assert per-(group) birth dates stay group-specific.
    shared = (set(df.loc[df.group == "meniscus", "anonyme"]) &
              set(df.loc[df.group == "cyclops", "anonyme"]))
    # For every shared id, birth date within a (group, id) is constant (min),
    # and the per-group hygienised value equals that group's own minimum.
    for aid in list(shared)[:5]:
        for grp in ("meniscus", "cyclops"):
            raw = df[(df.group == grp) & (df.anonyme == aid)]["date_de_naissance"]
            hyg = hy[(hy.group == grp) & (hy.anonyme == aid)]["date_de_naissance"]
            assert (hyg == raw.min()).all()


def test_wide_shape(wide):
    assert wide.shape[0] == N_TOTAL == 69
    assert "delta_lesion_total" in wide.columns
    assert "delta_lesion_pf" in wide.columns
    assert "delta_lesion_ft" in wide.columns
    assert "worsened_pf" in wide.columns
    assert "worsened_ft" in wide.columns
    assert "inter_surgery_d" in wide.columns


def test_analysable_pf_cohort_is_69(wide):
    """The progression cohort with a usable PF outcome is n = 69 (49 cyc + 20 mén).

    After patient 25 was reclassified cyclops→meniscus (2026-05-29) and its
    operated-today S2 data completed, no row is all-NaN any longer, so every
    patient carries a usable PF delta (consensus point §2.6, amended).
    """
    pf = wide.dropna(subset=["delta_lesion_pf"])
    assert pf.shape[0] == N_TOTAL_ANALYSABLE == 69
    by = pf.groupby("group").size().to_dict()
    assert by["cyclops"] == N_CYCLOPS_ANALYSABLE == 49
    assert by["meniscus"] == N_MENISCUS == 20


def test_female_indicator_from_sexe():
    """Sexe ∈ {1,2}; female = (sexe == 2). Anthropometric proof: 1 = M, 2 = F."""
    df = loaders.load_combined()
    df = pp.add_derived(df)
    # female is 1 exactly where sexe == 2
    s = pd.to_numeric(df["sexe"], errors="coerce")
    assert (df["female"].astype("Int64") == (s == 2).astype("Int64")).all()
    # Anthropometric sanity: Sexe=1 taller/heavier than Sexe=2 (so 1 = male).
    m1 = df[df["sexe"] == 1][["taille", "poids"]].mean()
    m2 = df[df["sexe"] == 2][["taille", "poids"]].mean()
    assert m1["taille"] > m2["taille"] and m1["poids"] > m2["poids"]


def test_pf_signal_supported(wide):
    """PF contrast reproduces the canonical large effect (δ ≈ 0.53, p ≈ 0.0002)."""
    res = tf.pf_contrast(wide, n_boot=2000, n_perm=5000, seed=RANDOM_SEED)
    assert 0.45 <= res["cliffs_delta"] <= 0.62, res["cliffs_delta"]
    assert res["mwu_p"] < 0.001
    assert res["perm_p"] < 0.005
    assert res["cliffs_delta_magnitude"] == "large"


def test_collapse_scores_maps_3_to_2():
    """collapse_scores maps grade 3 → 2 and leaves {0,1,2} and NaN intact."""
    fake = pd.DataFrame({
        "trochlée": pd.array([0, 1, 2, 3, pd.NA], dtype="Int64"),
        "rotule": pd.array([3, 3, 0, 1, 2], dtype="Int64"),
    })
    out = pp.collapse_scores(fake, ["trochlée", "rotule"])
    assert out["trochlée"].tolist()[:4] == [0, 1, 2, 2]
    assert pd.isna(out["trochlée"].iloc[4])
    assert out["rotule"].tolist() == [2, 2, 0, 1, 2]
    assert out["trochlée"].max() <= SCORE_MAX_COLLAPSED


def test_patient_shape(patient):
    assert patient.shape[0] == N_TOTAL == 69


def test_delta_range(wide):
    d = wide["delta_lesion_total"].dropna()
    assert d.between(-12, 12).all()


# --- tests_freq smoke ------------------------------------------------------

def test_mwu_with_effects_smoke():
    rng = np.random.default_rng(RANDOM_SEED)
    x = rng.normal(0, 1, 50)
    y = rng.normal(0.5, 1, 50)
    res = tf.mwu_with_effects(x, y, n_boot=200)
    assert "cliffs_delta" in res
    assert -1.0 <= res["cliffs_delta"] <= 1.0
    assert res["delta_ci_lo"] <= res["cliffs_delta"] <= res["delta_ci_hi"]


def test_wilcoxon_signed_rank_smoke():
    rng = np.random.default_rng(RANDOM_SEED)
    x = rng.normal(0, 1, 20)
    y = x + rng.normal(0.2, 0.2, 20)
    res = tf.wilcoxon_exact_with_rrb(y, x)
    assert "rank_biserial" in res
    assert -1.0 <= res["rank_biserial"] <= 1.0


def test_bh_fdr_smoke():
    pvals = [0.001, 0.02, 0.05, 0.5, 0.8]
    res = tf.bh_fdr(pvals, q=0.10)
    assert sum(res["reject"]) >= 2


def test_permutation_test_cliff_exact_small():
    """Exact permutation for a tiny clear separation returns a small p-value."""
    x = np.array([3.0, 4.0, 5.0])
    y = np.array([0.0, 1.0])
    res = tf.permutation_test_cliff(x, y, alternative="two-sided")
    assert res["method"] == "exact"  # C(5,3)=10 ≤ exact_max
    assert res["cliffs_delta"] == 1.0
    assert 0.0 < res["pvalue"] <= 0.2


def test_permutation_test_cliff_monte_carlo():
    """Large n forces Monte-Carlo; p-value uses (b+1)/(m+1) so never 0."""
    rng = np.random.default_rng(RANDOM_SEED)
    x = rng.normal(1.0, 1.0, 40)
    y = rng.normal(0.0, 1.0, 40)
    res = tf.permutation_test_cliff(x, y, n_perm=2000, seed=RANDOM_SEED)
    assert res["method"] == "monte-carlo"
    assert res["pvalue"] >= 1.0 / (2000 + 1)
    assert res["pvalue"] < 0.05


def test_cliff_ci_inversion_brackets_point():
    """Analytic inversion CI brackets the point δ and stays within ±1."""
    rng = np.random.default_rng(RANDOM_SEED)
    x = rng.normal(1.0, 1.0, 50)
    y = rng.normal(0.0, 1.0, 30)
    d = tf.cliffs_delta(x, y)
    lo, hi = tf.cliff_ci_inversion(x, y, ci=0.95)
    assert -1.0 <= lo <= d <= hi <= 1.0


def test_paired_pf_vs_ft_smoke(wide):
    """Within-cyclops ΔPF-vs-ΔFT Wilcoxon runs and returns rank-biserial."""
    res = tf.paired_pf_vs_ft(wide)
    assert res["cyclops"] == "cyclops"
    assert "rank_biserial" in res
    assert res["n_pf_worsened"] >= res["n_ft_worsened"]


def test_sensitivity_covariate_with_without(merged):
    """Sport/occupation sensitivity returns crude and adjusted odds ratios."""
    res = tf.sensitivity_covariate(merged, outcome_col="worsened_pf")
    assert "or_crude" in res and "or_adjusted" in res
    assert res["n_crude"] >= res["n_adjusted"]


# --- Constants -------------------------------------------------------------

def test_constants():
    assert SITES == ["trochlée", "rotule", "pte", "pti", "cfe", "cfi"]
    assert GROUPS == ["meniscus", "cyclops"]
    assert N_MENISCUS == 20
    assert N_CYCLOPS == 49
    assert N_TOTAL == 69
    assert N_TOTAL_ANALYSABLE == 69
    assert N_CYCLOPS_ANALYSABLE == 49
    assert RANDOM_SEED == 42
    assert (SCORE_MIN, SCORE_MAX) == (0, 3)
    assert SCORE_MAX_COLLAPSED == 2
    # Topographic blocks (corrected anatomy).
    assert SITES_PF == ["trochlée", "rotule"]
    assert SITES_FT == ["pte", "pti", "cfe", "cfi"]
    # Amended 2026-05-29: pte/cfe reclassed to Bernoulli (only 2 events ≥2 each
    # → upper cutpoint prior-driven); only rotule/trochlée stay ordinal.
    assert SITES_BINARY == ["pte", "pti", "cfe", "cfi"]
    assert SITES_ORDINAL == ["trochlée", "rotule"]
    assert BLOCKS == ["PF", "FT"]
    # Every site belongs to exactly one block; binary ∪ ordinal = all sites.
    assert set(SITES_PF) | set(SITES_FT) == set(SITES)
    assert set(SITES_BINARY) | set(SITES_ORDINAL) == set(SITES)
    assert set(SITES_BINARY) & set(SITES_ORDINAL) == set()


# --- Edge-case tests (pass 2) ----------------------------------------------

def test_load_handles_missing_data():
    """Site columns load as nullable Int64 (NaN would survive without float promotion).

    The single all-NaN cyclops row (patient 25, operated-today) was completed and
    reclassified to meniscus on 2026-05-29, so the site grid is now fully populated
    in BOTH groups. The invariant under test is the nullable-Int64 dtype that
    *guarantees* NaN-preservation if any reappears — not the (now zero) NaN count.
    """
    df = loaders.load_combined()
    # Cohort is now complete: no NaN in either group's site grid.
    cyclops = df[df["group"] == "cyclops"]
    meniscus = df[df["group"] == "meniscus"]
    for s in SITES:
        assert int(cyclops[s].isna().sum()) == 0, f"Cyclops unexpectedly NaN in {s}"
        assert int(meniscus[s].isna().sum()) == 0, f"Méniscus should have no NaN in {s}"
    # dtype must be nullable Int64 (preserves NaN without float promotion)
    for s in SITES:
        assert str(df[s].dtype) == "Int64", f"{s} dtype = {df[s].dtype}, expected Int64"


def test_delta_total_for_unchanged_patient(wide):
    """A patient with identical S1 and S2 site scores must have delta_total == 0."""
    # Synthetic check via direct computation on a minimal frame
    fake = pd.DataFrame({
        "group": ["meniscus", "meniscus"],
        "anonyme": [999, 999],
        "surgery_num": pd.array([1, 2], dtype="Int64"),
        "date_chir": pd.to_datetime(["2024-01-01", "2024-06-01"]),
        "date_du_trauma": pd.to_datetime(["2023-01-01", "2023-01-01"]),
        **{s: pd.array([1, 1], dtype="Int64") for s in SITES},
    })
    fake = pp.add_derived(fake)
    fake_wide = pp.to_wide(fake)
    assert int(fake_wide["delta_lesion_total"].iloc[0]) == 0, (
        f"Expected delta_total == 0 for identical S1/S2, got {fake_wide['delta_lesion_total'].iloc[0]}"
    )
    for s in SITES:
        col = f"delta_{s}"
        assert int(fake_wide[col].iloc[0]) == 0, f"Expected delta_{s} == 0"

    # Also: in the real cohort, any patient with delta_lesion_total == 0 should
    # have all six per-site deltas == 0.
    zero_patients = wide[wide["delta_lesion_total"] == 0]
    for _, row in zero_patients.iterrows():
        site_deltas = [row[f"delta_{s}"] for s in SITES if f"delta_{s}" in wide.columns]
        # Allow some patients with NaN in a site (cyclops): only assert on non-NaN
        non_na = [d for d in site_deltas if pd.notna(d)]
        if non_na:
            assert sum(non_na) == 0, (
                f"Patient {row['anonyme']} has delta_total=0 but per-site deltas sum to {sum(non_na)}"
            )


def test_idempotent_load():
    """Two consecutive loads must return identical data."""
    df1 = loaders.load_combined()
    df2 = loaders.load_combined()
    # Columns equal, shapes equal, values equal (NaN-aware via pandas.testing)
    pd.testing.assert_frame_equal(df1, df2, check_like=False)


def test_normalised_cols_idempotent():
    """Applying _normalise_cols twice yields the same columns as once."""
    df = loaders.load_meniscus()  # already normalised
    cols_once = list(df.columns)
    df2 = _normalise_cols(df)
    cols_twice = list(df2.columns)
    assert cols_once == cols_twice, (
        f"_normalise_cols not idempotent.\nfirst: {cols_once}\nsecond: {cols_twice}"
    )


def test_summary_bayes_api():
    """``summary_bayes`` calls the modern ArviZ API (ci_prob/ci_kind) without error.

    Smoke-tests on a tiny synthetic InferenceData (no PyMC sampling required).
    """
    import arviz as az

    rng = np.random.default_rng(RANDOM_SEED)
    # Synthetic posterior: 2 chains × 200 draws × 1 var (scalar) + 1 var (3-dim).
    posterior = {
        "delta": rng.normal(0.3, 0.4, size=(2, 200)),
        "beta_c": rng.normal(0.0, 0.5, size=(2, 200, 3)),
    }
    # ArviZ ≥ 1.0 dropped the legacy ``posterior=...`` kw in favour of a
    # nested dict ``{group: {var: array}}`` passed positionally.
    try:
        idata = az.from_dict(
            {"posterior": posterior},
            coords={"comp": ["c0", "c1", "c2"]},
            dims={"beta_c": ["comp"]},
        )
    except TypeError:
        # Legacy ArviZ (< 1.0) — keyword API
        idata = az.from_dict(  # type: ignore[call-arg]
            posterior=posterior,
            coords={"comp": ["c0", "c1", "c2"]},
            dims={"beta_c": ["comp"]},
        )
    summ = rpt.summary_bayes(idata, var_names=["delta", "beta_c"], hdi_prob=0.94)
    assert "P(>0)" in summ.columns, f"Expected P(>0) column, got {list(summ.columns)}"
    # delta has true mean 0.3 so P(>0) should be roughly > 0.7
    p_delta = summ.loc["delta", "P(>0)"]
    assert 0.6 <= p_delta <= 1.0, f"P(delta > 0) = {p_delta} out of expected range."
    # Has at least one HDI-style column (modern or legacy)
    hdi_cols = [c for c in summ.columns if "hdi" in c.lower()]
    assert hdi_cols, f"No HDI columns in summary: {list(summ.columns)}"


def test_bca_bootstrap_seed_determinism():
    """Two BCa runs with the same seed must produce identical CIs."""
    rng = np.random.default_rng(RANDOM_SEED)
    x = rng.normal(0, 1, 40)
    y = rng.normal(0.4, 1, 40)
    r1 = tf.mwu_with_effects(x, y, n_boot=400, seed=RANDOM_SEED)
    r2 = tf.mwu_with_effects(x, y, n_boot=400, seed=RANDOM_SEED)
    assert r1["delta_ci_lo"] == r2["delta_ci_lo"], (
        f"BCa lo not deterministic: {r1['delta_ci_lo']} vs {r2['delta_ci_lo']}"
    )
    assert r1["delta_ci_hi"] == r2["delta_ci_hi"], (
        f"BCa hi not deterministic: {r1['delta_ci_hi']} vs {r2['delta_ci_hi']}"
    )
    # Different seed should (generally) give different CI
    r3 = tf.mwu_with_effects(x, y, n_boot=400, seed=RANDOM_SEED + 1)
    assert (r1["delta_ci_lo"], r1["delta_ci_hi"]) != (r3["delta_ci_lo"], r3["delta_ci_hi"]), (
        "BCa CI suspiciously identical across two different seeds."
    )


# --- Interpretability helpers (code-review high pass) ----------------------

def test_interpret_cliffs_delta_thresholds():
    """Romano (2006) verbal magnitudes around boundary values."""
    assert tf.interpret_cliffs_delta(0.0) == "negligible"
    assert tf.interpret_cliffs_delta(0.10) == "negligible"
    assert tf.interpret_cliffs_delta(0.20) == "small"
    assert tf.interpret_cliffs_delta(0.40) == "medium"
    assert tf.interpret_cliffs_delta(0.60) == "large"
    # sign-invariant
    assert tf.interpret_cliffs_delta(-0.40) == "medium"


def test_format_test_result_mwu_shape():
    """format_test_result returns a non-empty French sentence with stats + magnitude."""
    fake = dict(
        statistic=349.5, pvalue=0.088,
        cliffs_delta=-0.25, delta_ci_lo=-0.48, delta_ci_hi=0.03,
        cliffs_delta_magnitude="small", probability_of_superiority=0.38,
        n_x=19, n_y=50,
    )
    sentence = rpt.format_test_result(fake, "mwu")
    assert isinstance(sentence, str) and len(sentence) > 30
    for token in ("Mann-Whitney", "Cliff", "small", "P(X > Y)"):
        assert token in sentence, f"missing {token!r} in: {sentence}"


def test_interpret_bayes_rope_pct():
    """interpret_bayes ROPE % is high when posterior is centred in ROPE."""
    import arviz as az

    rng = np.random.default_rng(RANDOM_SEED)
    posterior_centred = {"beta": rng.normal(0.0, 0.03, size=(2, 400))}
    posterior_shifted = {"beta": rng.normal(0.5, 0.05, size=(2, 400))}
    try:
        idata_c = az.from_dict({"posterior": posterior_centred})
        idata_s = az.from_dict({"posterior": posterior_shifted})
    except TypeError:
        idata_c = az.from_dict(posterior=posterior_centred)  # type: ignore[call-arg]
        idata_s = az.from_dict(posterior=posterior_shifted)  # type: ignore[call-arg]
    out_c = rpt.interpret_bayes(idata_c, "beta", rope=(-0.1, 0.1))
    out_s = rpt.interpret_bayes(idata_s, "beta", rope=(-0.1, 0.1))
    assert out_c["rope_pct"] > 80.0, f"centred posterior ROPE% = {out_c['rope_pct']}"
    assert out_s["rope_pct"] < 5.0, f"shifted posterior ROPE% = {out_s['rope_pct']}"
    assert out_s["hdi_excludes_rope"] is True
    assert "narrative" in out_c and "narrative" in out_s


def test_summary_bayes_scalar_var_no_crash():
    """summary_bayes must not raise on a purely scalar posterior (chain × draw)."""
    import arviz as az

    rng = np.random.default_rng(RANDOM_SEED)
    posterior = {"gamma": rng.normal(0.0, 1.0, size=(2, 300))}
    try:
        idata = az.from_dict({"posterior": posterior})
    except TypeError:
        idata = az.from_dict(posterior=posterior)  # type: ignore[call-arg]
    summ = rpt.summary_bayes(idata, var_names=["gamma"], hdi_prob=0.94)
    assert "P(>0)" in summ.columns
    assert "gamma" in summ.index


def test_lesion_total_strict_nan_when_site_missing():
    """``lesion_total`` is STRICT (NaN if any site NaN); permissive kept aside.

    Consensus point I.5: the analysis ``lesion_total`` is NaN if any
    compartment is missing (``min_count = len(sites)``). The permissive
    "sum of present" variant is retained as ``lesion_total_permissive``.
    """
    fake = pd.DataFrame({
        "group": ["meniscus", "meniscus"],
        "anonyme": [9999, 9999],
        "surgery_num": pd.array([1, 2], dtype="Int64"),
        "date_chir": pd.to_datetime(["2024-01-01", "2024-06-01"]),
        "date_du_trauma": pd.to_datetime(["2023-01-01", "2023-01-01"]),
        # 5 sites = 1 each, 6th = NaN at S1 → strict total = NaN, permissive = 5
        "trochlée": pd.array([1, 1], dtype="Int64"),
        "rotule":   pd.array([1, 1], dtype="Int64"),
        "pte":      pd.array([1, 1], dtype="Int64"),
        "pti":      pd.array([1, 1], dtype="Int64"),
        "cfe":      pd.array([1, 1], dtype="Int64"),
        "cfi":      pd.array([pd.NA, 1], dtype="Int64"),
    })
    fake = pp.add_derived(fake)
    s1 = fake.loc[fake["surgery_num"] == 1].iloc[0]
    s2 = fake.loc[fake["surgery_num"] == 2].iloc[0]
    assert pd.isna(s1["lesion_total"]), "strict total must be NaN when any site NaN"
    assert int(s2["lesion_total"]) == 6
    assert int(s1["lesion_total_permissive"]) == 5
    assert int(s1["lesion_total_strict"]) == s1["lesion_total"] if pd.notna(s1["lesion_total"]) else pd.isna(s1["lesion_total_strict"])
    assert int(s1["n_sites_observed"]) == 5
    assert int(s2["n_sites_observed"]) == 6
    # PF block sum present at both times (both PF sites observed).
    assert int(s1["lesion_pf"]) == 2 and int(s2["lesion_pf"]) == 2
    # FT block: cfi NaN at S1 → strict FT NaN at S1.
    assert pd.isna(s1["lesion_ft"]) and int(s2["lesion_ft"]) == 4


def test_bca_zero_variance_warns():
    """Constant input → jackknife zero variance → RuntimeWarning emitted once."""
    x = np.zeros(20)  # constant array makes Cliff's δ constant → zero jackknife var
    y = np.zeros(20)
    with pytest.warns(RuntimeWarning, match="BCa acceleration undefined"):
        tf.mwu_with_effects(x, y, n_boot=200, seed=RANDOM_SEED)


# --- Revue 2026-05-29 additions (A–E) --------------------------------------

def test_detect_date_anomalies_flags_known():
    """detect_date_anomalies (on RAW frame) flags #9 drift and #38 negative delay."""
    df = loaders.load_combined()
    an = pp.detect_date_anomalies(df)
    assert {"group", "anonyme", "kind", "detail"}.issubset(an.columns)
    negs = an[an["kind"] == "negative_trauma_to_surgery_d"]
    assert not negs.empty and 38 in set(negs["anonyme"]), an.to_dict("records")
    # #9 trauma date drifts ~2 months between S1 and S2 (a drift anomaly).
    drifts = an[an["kind"].str.contains("drift")]
    assert 9 in set(drifts["anonyme"]), an.to_dict("records")


def test_negative_delay_coerced_to_nan():
    """add_derived coerces impossible (negative) trauma_to_surgery_d to NaN."""
    df = loaders.load_combined()
    df = pp.apply_date_hygiene(df)
    df = pp.add_derived(df)
    assert (df["trauma_to_surgery_d"].dropna() >= 0).all()


def test_firth_or_finite_under_separation(wide):
    """Firth OR is finite & bracketed under the 1/20 meniscus quasi-separation."""
    res = tf.firth_or(wide, outcome_col="worsened_pf")
    assert np.isfinite(res["odds_ratio"]) and res["odds_ratio"] > 1.0
    assert res["or_ci_lo"] < res["odds_ratio"] < res["or_ci_hi"]
    assert res["min_cell"] <= 1  # quasi-separation (single meniscus PF event)


def test_evalue_or_monotone_and_null_crossing():
    """E-value grows with OR; a CI bound crossing the null gives E-value 1."""
    e_small = tf.evalue_or(2.0)["evalue_point"]
    e_big = tf.evalue_or(16.0)["evalue_point"]
    assert e_big > e_small > 1.0
    assert tf.evalue_or(3.0, or_ci_lo=0.8)["evalue_ci"] == 1.0


# --- Baseline balance & equivalence (MWU + SMD + TOST) ---------------------
# All three S1 levels — global 6-sum, PF block, FT block — go through the SAME
# baseline_block_balance (MWU difference + SMD magnitude + TOST equivalence).
# Only PF is positively equivalent; the 6-sum and FT are non-significant on MWU
# yet fail TOST at n = 20 (absence of evidence is not evidence of absence).

def test_tost_equivalence_basic():
    """TOST: equal samples are equivalent within a modest bound; far apart aren't."""
    rng = np.random.default_rng(RANDOM_SEED)
    eq = tf.tost_equivalence(rng.normal(0, 1, 60), rng.normal(0, 1, 60), bound=0.8)
    assert 0.0 <= eq["tost_p"] <= 1.0
    far = tf.tost_equivalence(rng.normal(0, 1, 60), rng.normal(3, 1, 60), bound=0.5)
    assert far["tost_p"] > 0.05  # cannot conclude equivalence


def test_smd_continuous_sign():
    """smd_continuous: sign follows (x − y), magnitude scales with the gap.

    Dedicated cover for the SMD primitive shared by every baseline check
    (previously exercised only implicitly via the balance dicts)."""
    rng = np.random.default_rng(RANDOM_SEED)
    y = rng.normal(0.0, 1.0, 200)
    assert abs(tf.smd_continuous(rng.normal(0.0, 1.0, 200), y)) < 0.3   # same dist → ~0
    assert tf.smd_continuous(rng.normal(1.0, 1.0, 200), y) > 0.5        # x up → +large
    assert tf.smd_continuous(y, rng.normal(1.0, 1.0, 200)) < -0.5       # antisymmetric


def test_baseline_pf_balance_keys(wide):
    """baseline_pf_balance returns MWU p, SMD, and a TOST equivalence verdict."""
    res = tf.baseline_pf_balance(wide)
    for k in ("mwu_p", "smd", "tost_p", "equivalent", "tost_bound"):
        assert k in res


def test_baseline_total_equivalent_keys(wide):
    """The global 6-sum now gets the SAME unified treatment as the blocks.

    Confirms baseline_block_balance runs on lesion_total_S1 and returns the full
    key schema. Like FT (and unlike PF), the 6-sum is non-significant on MWU yet
    NOT established as equivalent by TOST at n = 20."""
    res = tf.baseline_block_balance(wide, col="lesion_total_S1")
    for k in ("mwu_p", "smd", "tost_p", "tost_bound", "equivalent",
              "median_cyclops", "median_meniscus"):
        assert k in res
    assert res["mwu_p"] > 0.05            # no difference *detected* overall


def test_baseline_block_balance_ft_not_equivalent(wide):
    """FT block is NOT proven equivalent at S1 — the 'absence of evidence is not
    evidence of absence' case that motivates testing FT symmetrically to PF.

    MWU is non-significant (no *detected* difference) yet TOST fails (equivalence
    *not established*): cyclops start with a small but non-negligible excess of
    femorotibial lesion (positive SMD). This is the baseline gap that makes the
    PF−FT topographic contrast exploratory rather than cleanly causal.
    """
    ft = tf.baseline_block_balance(wide, col="lesion_ft_S1")
    assert ft["col"] == "lesion_ft_S1"
    assert ft["mwu_p"] > 0.05            # no difference *detected*
    assert ft["equivalent"] is False     # ...but equivalence NOT established
    assert ft["smd"] > 0.147             # at least a "small" imbalance (cyclops higher)


# --- Bayesian model structure ----------------------------------------------

def test_pooling_grouping_structures():
    """_pooling_grouping yields 1/2/3 groups covering all six sites."""
    import bayes_models as bm
    for pool, n in (("exchangeable", 1), ("two_block", 2), ("three_cluster", 3)):
        names, idx = bm._pooling_grouping(pool, SITES)
        assert len(names) == n and len(idx) == len(SITES)
        assert set(int(i) for i in idx) == set(range(n))
    with pytest.raises(ValueError):
        bm._pooling_grouping("nonsense", SITES)


def test_build_m3_three_poolings(df_long):
    """All three pooling structures build and expose the right named estimands."""
    import bayes_models as bm
    long = bm._melt_long_long(df_long, SITES, "anonyme", "group", collapse=True)
    want = {"exchangeable": "delta_bar", "two_block": "contrast_pf_ft"}
    for pool in ("exchangeable", "two_block", "three_cluster"):
        m = bm._build_m3_model(long, SITES, pooling=pool)
        det = {d.name for d in m.deterministics}
        assert "delta_comp" in det, f"{pool}: missing delta_comp"
        if pool in want:
            assert want[pool] in det, f"{pool}: missing {want[pool]}"
