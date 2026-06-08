"""lino_stats  Dual-population bayesian analysis of paired knee-surgery cohort.

Shared constants and lazy interpretability re-exports for the flat ``src/``
module layout (each analysis module  loaders, preprocessing, viz, tests_freq,
bayes_models, reporting  is a top-level module on ``pythonpath = ["src"]``).

Modules
-------
loaders : read Excel sheets and normalise column names.
preprocessing : data hygiene, derived variables, wide/long pivots.
viz : seaborn / plotly figures (paired slopegraph, Sankey, heatmap, ...).
tests_freq : non-parametric tests with effect sizes and bootstrap CIs.
bayes_models : PyMC models M1-M5 (beta-binomial, negbin, hierarchical ordinal, AFT, lognormal).
reporting : Table 1, ArviZ summaries, markdown export.
"""

__version__ = "0.1.0"

# --- Constants (locked by 99-Project-Log/contract.md) -----------------------
RANDOM_SEED: int = 42

# Compartments lésionnels (lowercase, accents preserved per contract §1)
SITES: list[str] = ["trochlée", "rotule", "pte", "pti", "cfe", "cfi"]

# --- Topographic blocks (revue-methodo-2026-05 §2.2, consensus point B) -----
# Corrected anatomy: PTE/PTI are *tibial plateaus*, not patellofemoral.
#   PF (patellofemoral) = {trochlée, rotule}     ← carries the entire signal
#   FT (femorotibial)   = {pte, pti, cfe, cfi}
# The PRIMARY estimand is the PF contrast (cyclops vs meniscus) and the
# mechanistic co-primary μ_PF − μ_FT (M3 two-block pooling, point E).
SITES_PF: list[str] = ["trochlée", "rotule"]
SITES_FT: list[str] = ["pte", "pti", "cfe", "cfi"]

# --- Per-site measurement model (revue-methodo 2026-05-29, point E1 amended) -
# Likelihood matched to the OBSERVED event counts ≥2 (recomputed on the data):
#   rotule   n(≥2)=15 → ordinal cumulative logit {0,1,≥2} (2 cutpoints, identified)
#   trochlée n(≥2)=3  → ordinal (marginal but admissible)
#   pte/cfe  n(≥2)=2  → upper cutpoint would be PRIOR-driven → Bernoulli {0,≥1}
#   pti/cfi  n(≥2)=0  → strictly Bernoulli {0,≥1}
# Reclassing pte/cfe from ordinal to binary (vs the original 2026-05 spec)
# aligns the likelihood with what 2 stray events can actually identify; only
# rotule (and marginally trochlée) supports an ordinal upper cutpoint.
SITES_BINARY: list[str] = ["pte", "pti", "cfe", "cfi"]
SITES_ORDINAL: list[str] = ["trochlée", "rotule"]

# Mapping site → topographic block index used by M3 (0 = PF, 1 = FT).
BLOCKS: list[str] = ["PF", "FT"]
SITE_BLOCK: dict[str, int] = {
    **{s: 0 for s in SITES_PF},
    **{s: 1 for s in SITES_FT},
}

# Group labels (snake_case english per contract §1)
GROUPS: list[str] = ["meniscus", "cyclops"]

# Cohort sizes (locked). 2026-05-29: patient anonyme=25 reclassified
# cyclops→meniscus (clinically a meniscus patient; its operated-today S2 data
# arrived), so cyclops 50→49 and meniscus 19→20; total unchanged at 69.
N_MENISCUS: int = 20
N_CYCLOPS: int = 49
N_TOTAL: int = 69

# Analysable progression cohort (revue-methodo-2026-05 §2.6): after the patient-25
# reclassification every patient carries a usable PF outcome → 69 analysable
# (49 cyclops + 20 meniscus). The former all-NaN cyclops row was patient 25,
# whose operated-today S2 data is now complete. Canonical denominator in run_all.
N_CYCLOPS_ANALYSABLE: int = 49
N_TOTAL_ANALYSABLE: int = 69

# Row counts (long format, 2 rows per patient)
N_MENISCUS_ROWS: int = 40
N_CYCLOPS_ROWS: int = 98
N_ROWS: int = 138

# Ordinal score bounds. NATIVE scale is 0–3 (Outerbridge grade-3 in 2 rows on
# trochlée/rotule)  kept for the descriptive raw-frequency table and
# sensitivity analyses. For MODELLING (M3) the scale is COLLAPSED to {0,1,≥2}
# (consensus point A): grade 3 → 2. The collapse is information-neutral on the
# PF signal (§2.2: PF δ 0.533 → 0.532) and clinically justified (ICRS/Outerbridge
# grades 2 = deep loss and 3 = exposed bone form one "advanced lesion" pole).
SCORE_MIN: int = 0
SCORE_MAX: int = 3  # native scale (descriptive)
SCORE_MAX_COLLAPSED: int = 2  # collapsed {0,1,≥2} scale (modelling)

# Sensitivity priors on β_c (contract §4.3, M3)  pass 2 sensitivity grid.
SENSITIVITY_PRIOR_SIGMAS: tuple[float, ...] = (0.5, 1.0, 5.0)

# HDI mass (Kruschke convention, contract §2)
HDI_PROB: float = 0.94

# Bootstrap defaults (BCa CI). Raised 5000 → 10000 per consensus point H
# (at n_mén = 19 the limiting factor is n, not B; we still report a
# permutation-inversion CI as a guard-rail alongside BCa).
N_BOOT_DEFAULT: int = 10000
CI_DEFAULT: float = 0.95

# Equivalence (TOST) margin for baseline-balance checks. The equivalence box is
# ±(EQUIV_SMD_MARGIN × pooled SD) on the raw scale  i.e. an SMD margin of this
# value. 0.5 = Cohen's "small" boundary: a difference below half a pooled SD is
# treated as clinically negligible. Single source of truth for the TOST bound in
# baseline_block_balance (was hardcoded inline before).
EQUIV_SMD_MARGIN: float = 0.5

# Permutation test defaults (consensus point H): exact if the number of
# group-label assignments is tractable, else Monte-Carlo with this many
# resamples. Seed fixed for reproducibility.
N_PERM_DEFAULT: int = 20000
PERM_EXACT_MAX: int = 1_000_000  # max C(n, k) before falling back to Monte-Carlo

# Convergence thresholds (contract §3, locked)
RHAT_MAX: float = 1.01
ESS_MIN: float = 400

# NUTS defaults (contract §3, §7)
NUTS_KWARGS: dict = dict(
    chains=4,
    tune=2000,
    draws=2000,
    target_accept=0.95,
    random_seed=RANDOM_SEED,
    progressbar=False,
)

# NUTS escalation kwargs (apply when divergences > 0 or r_hat > RHAT_MAX)
NUTS_KWARGS_ESCALATED: dict = dict(
    chains=4,
    tune=4000,
    draws=2000,
    target_accept=0.99,
    random_seed=RANDOM_SEED,
    progressbar=False,
)


# --- Public interpretability helpers (append-only re-exports) --------------


def __getattr__(name: str):
    """Lazy re-export of sibling-module helpers (avoids eager imports / cycles)."""
    _lazy_map = {
        "interpret_cliffs_delta": ("tests_freq", "interpret_cliffs_delta"),
        "interpret_rank_biserial": ("tests_freq", "interpret_rank_biserial"),
        "interpret_cohens_d": ("tests_freq", "interpret_cohens_d"),
        "smd_with_magnitude": ("tests_freq", "smd_with_magnitude"),
        "permutation_test_cliff": ("tests_freq", "permutation_test_cliff"),
        "pf_contrast": ("tests_freq", "pf_contrast"),
        "paired_pf_vs_ft": ("tests_freq", "paired_pf_vs_ft"),
        "format_test_result": ("reporting", "format_test_result"),
        "interpret_bayes": ("reporting", "interpret_bayes"),
        "verdict_bayes_en": ("reporting", "verdict_bayes_en"),
        "verdict_freq_en": ("reporting", "verdict_freq_en"),
        "make_clinical_summary_card": ("viz", "make_clinical_summary_card"),
        "slopegraph_paired_annotated": ("viz", "slopegraph_paired_annotated"),
        "sankey_transitions_normalized": ("viz", "sankey_transitions_normalized"),
        "annotate_forest_clinical_bounds": ("viz", "annotate_forest_clinical_bounds"),
        "posterior_risk_at_S2": ("bayes_models", "posterior_risk_at_S2"),
    }
    if name in _lazy_map:
        import importlib

        mod_name, attr = _lazy_map[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
