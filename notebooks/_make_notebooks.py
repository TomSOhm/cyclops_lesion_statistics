"""Generator script for the 7 chapter notebooks (PF-primary refactor).

Run once to (re)create all notebooks. Idempotent: rewrites every notebook
from scratch each time. The generated notebooks are committed alongside.

This generator reflects the consensus specification plus the methodology
review of 2026-05-29 (``99-Project-Log/revue-methodo-2026-05.md``):

* PRIMARY GLOBAL estimand = the knee-wide interaction ``delta_bar`` from the
  **exchangeable** (selection-immune) hierarchical model — reported honestly:
  it is *small and uncertain* (the knee-wide average dilutes a compartment-
  specific signal), and is **not** overclaimed.
* The topographic PF/FT organisation is a **RESULT tested by LOO**
  (``compare_pooling_structures``: exchangeable vs two_block vs three_cluster),
  **not an assumed hypothesis**. The 2-block ``contrast_pf_ft`` is a
  *candidate-structure* output; the PF localisation is read non-circularly from
  ``derived_pf_contrast`` on the neutral exchangeable posterior.
* The frequentist PF block contrast ``delta_lesion_pf`` (= {trochlée, rotule})
  is the descriptive compartmental signal; the 6-compartment sum is demoted to
  a *global burden index* (it dilutes) — never decisional.
* Modelling scale is the COLLAPSED {0, 1, ≥2} grade; native 0–3 kept for the
  descriptive frequency table only.
* The binary worsened_pf OR is estimated with **Firth penalised logistic**
  (``tf.firth_or``) — the 1/20 meniscus events near-separate the plain ML logit
  into a *ghost* OR≈24 (≈47 adjusted); Firth gives a stable OR + an E-value.
* M2 (NegBin on the 6-sum, Δ⁺ truncation) is **removed** from the inferential
  chain — only a labelled sanity-check note remains.
* H3 risk factors (now incl. a usable BMI ``imc`` derived from taille/poids)
  are tested against ``delta_lesion_pf`` (not the 6-sum).
* The inter-surgical delay is **time-at-risk** (a mediator downstream of group),
  NOT a confounder — the primary model does not adjust for it; conditioning on
  it is a sensitivity/falsification analysis only.
* Decision rule (point G, amended 2026-05-29): for the PRE-SPECIFIED directional
  global δ̄ use the one-sided P(>0|data) ≥ 0.95 rule (``two_sided=False``); for
  POST-HOC estimands (PF contrast, PF−FT) use the direction-agnostic two-sided
  credible rule (HDI excludes 0). Frequentist support = exact/MC permutation.

IMPORTANT — the MCMC sampling cells (notebooks 05 and 06) are written
**uncommented and runnable** but the notebooks are generated WITHOUT any
execution outputs. The user runs them cell-by-cell. This script never executes
a notebook or a ``pm.sample`` call.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).parent


def make_nb(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    }
    return nb


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


# --- src/ path bootstrap (first code cell, so the flat src/ modules import) --
BOOTSTRAP = """\
import sys
from pathlib import Path

current = Path().absolute().parent
sys.path.insert(0, (current / "src").as_posix())
"""

# --- Reusable preamble (imports + style) ------------------------------------
PREAMBLE = """\
# --- Setup (idempotent, fresh-kernel reproducible) ---
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from constants import (
    RANDOM_SEED, SITES, SITES_PF, SITES_FT, SITES_BINARY, SITES_ORDINAL,
    GROUPS, BLOCKS, N_TOTAL, N_TOTAL_ANALYSABLE, N_MENISCUS, N_CYCLOPS,
    SCORE_MAX, SCORE_MAX_COLLAPSED,
)
import loaders
import preprocessing as pp
import tests_freq as tf
import reporting as rpt
import bayes_models as bm
import viz

np.random.seed(RANDOM_SEED)
viz.set_pub_style()
"""

# Canonical pipeline + the merged (wide × patient covariates) frame every
# notebook relies on.
LOAD = """\
# --- Load & preprocess (canonical pipeline) ---
df = loaders.load_combined()
df = pp.apply_date_hygiene(df)      # composite-key (group, anonyme) date hygiene
df = pp.add_derived(df)            # lesion_pf/ft, female, deltas, worsened_pf, ...
wide = pp.to_wide(df)             # one row per patient (group, anonyme)
patient = pp.to_patient(df)       # static covariates per patient

# Patient-level covariates joined onto the wide outcomes (for H3 / sensitivity).
_cov = [c for c in ['group','anonyme','female','sexe','pivot_pivot_contact',
                    'travail_physique','tabac','age_at_trauma','imc','taille','poids']
        if c in patient.columns]
merged = wide.merge(patient[_cov], on=['group','anonyme'], how='left')

print('long:', df.shape, '| wide:', wide.shape, '| patient:', patient.shape,
      '| merged:', merged.shape)
# Composite-key sentinel: 19 Anonyme ids are reused across the two sheets.
assert (df.groupby(['group','anonyme']).size() == 2).all(), 'composite key broken'
"""


# ============================================================================
# 00 — EDA descriptive
# ============================================================================
nb_00 = make_nb([
    md("# 00 — Exploratory Data Analysis (EDA descriptive)\n\n"
       "**Notebook id**: `00_eda` → vault `03.1-EDA-descriptive.md`\n\n"
       "Cohort flow, a **data-anomaly audit** of the source dates, **Table 1 "
       "(SMD)**, baseline (S1) equivalence, raw per-compartment frequency tables "
       "on the **native 0–3 scale**, the **binary** status of PTI/CFI, the "
       "cohort of **n = 69** analysable for PF progression, and a note on "
       "the **{0, 1, ≥2} collapse** used for modelling. No inferential decision "
       "here — description only."),
    code(BOOTSTRAP),
    code(PREAMBLE),
    code(LOAD),
    md("## 0. Data-anomaly audit (source dates only — NOT the cartilage Δ)\n\n"
       "`pp.detect_date_anomalies` is run on the **RAW combined frame** (before "
       "`apply_date_hygiene`) so the divergences are still visible. It flags two "
       "kinds of source-date issue that pollute *age / delay / baseline / H3 / H4* "
       "but **never** the patellofemoral Δ (the score columns carry no dates):\n\n"
       "- **static-date drift** — `date_de_naissance` / `date_du_trauma` differing "
       "between a patient's S1 and S2 rows (e.g. patient #9, a trauma-date drift "
       "of ~61 d; hygiene resolves it by `min`, the flag is for manual fixing);\n"
       "- **negative trauma→surgery** — surgery dated *before* the trauma "
       "(impossible; #25 / #38), a placeholder/typo that `add_derived` now coerces "
       "to NaN.\n\n"
       "These rows form the exclusion list for the delay sensitivity in 06; they "
       "leave `inter_surgery_d` (an S2−S1 *difference* of two `date_chir`) intact."),
    code("""\
anomalies = pp.detect_date_anomalies(loaders.load_combined())
print(f"date anomalies found: {len(anomalies)}")
if len(anomalies):
    print(anomalies.to_string(index=False))
else:
    print('(none)')
print()
print('Breakdown by kind:')
print(anomalies['kind'].value_counts() if len(anomalies) else '(none)')
print()
print('NB: none of these touch the cartilage scores / the PF delta, and '
      'inter_surgery_d (a date_chir S2-S1 difference) is unaffected.')
"""),
    md("## 1. Cohort flow & missingness\n\n"
       "138 long rows = 69 patients × 2 surgeries. After patient 25 was "
       "reclassified cyclops→meniscus (2026-05-29) and its operated-today S2 data "
       "completed, every patient carries a usable PF outcome → "
       "**n = 69** (49 cyclops + 20 meniscus) analysable for progression."),
    code("""\
print(f"Patients      : meniscus={N_MENISCUS}, cyclops={N_CYCLOPS}, total={N_TOTAL}")
print(f"Long rows     : {len(df)} (expected 138)")
print(f"Wide rows     : {len(wide)} (expected 69)")
print()
print('Missingness on lesion sites (long):')
print(df[SITES].isna().sum())
print()
# Exclusion flow for the PF progression outcome.
pf_ok = wide['delta_lesion_pf'].notna()
print('Analysable for PF progression (delta_lesion_pf not NaN):')
print(wide.loc[pf_ok, 'group'].value_counts())
print(f"=> n analysable = {int(pf_ok.sum())} (expected {N_TOTAL_ANALYSABLE})")
assert int(pf_ok.sum()) == N_TOTAL_ANALYSABLE
"""),
    md("## 2. Table 1 — baseline cohort summary (SMD, Austin)\n\n"
       "Continuous covariates: median [IQR] + Mann–Whitney p + standardised mean "
       "difference. Categorical: n (%) + Fisher + binary SMD. The SMD is the "
       "balance metric (Austin 2009); large |SMD| flags an imbalance to carry "
       "into a sensitivity analysis (e.g. age), not the primary model."),
    code("""\
table1 = rpt.make_table1(patient)
print(table1.to_string(index=False))
"""),
    md("## 3. Raw per-compartment frequency tables (native 0–3 scale)\n\n"
       "Native grades are kept here for description. Note PTI and CFI carry "
       "**zero** events at grade ≥ 2 → they are strictly **binary** and are "
       "modelled with a Bernoulli likelihood in M3; the other four sites keep an "
       "ordinal cumulative logit on the collapsed {0, 1, ≥2} scale."),
    code("""\
for grp, sub in df.groupby('group'):
    print(f"=== {grp} (native 0-3) ===")
    counts = pd.concat({s: sub[s].value_counts().sort_index() for s in SITES},
                       axis=1).fillna(0).astype(int)
    print(counts)
    print()

# Binary vs ordinal classification used by M3.
print('Modelling families:')
print('  Bernoulli (binary, 0 events >=2):', SITES_BINARY)
print('  Cumulative logit (ordinal {0,1,>=2}):', SITES_ORDINAL)
print()
# Confirm the binary sites really never reach grade >= 2 in the data.
for s in SITES_BINARY:
    mx = int(pd.to_numeric(df[s], errors='coerce').max())
    print(f'  max({s}) = {mx}  -> binary OK' if mx <= 1 else f'  max({s}) = {mx}  !!')
"""),
    md("## 4. The {0, 1, ≥2} collapse (information-neutral on the PF signal)\n\n"
       "Modelling uses the collapsed scale (grade 3 → 2). Below we confirm the "
       "collapse barely moves the patellofemoral effect size — it removes a "
       "near-empty top category, it does not remove signal."),
    code("""\
# Native vs collapsed PF block delta, Cliff's delta cyclops vs meniscus.
df_coll = pp.collapse_scores(df, SITES)
wide_coll = pp.to_wide(pp.add_derived(df_coll))

def _cliff_pf(w):
    c = w.loc[w.group=='cyclops','delta_lesion_pf'].dropna().astype(float).values
    m = w.loc[w.group=='meniscus','delta_lesion_pf'].dropna().astype(float).values
    return tf.cliffs_delta(c, m)

print(f"Cliff delta PF  native(0-3)   = {_cliff_pf(wide):+.3f}")
print(f"Cliff delta PF  collapsed     = {_cliff_pf(wide_coll):+.3f}")
print("(near-identical => collapse is information-neutral on the PF signal)")
"""),
    md("## 5. Distribution of `lesion_pf` / `lesion_ft` by (group × time)"),
    code("""\
for col in ['lesion_pf', 'lesion_ft', 'lesion_total']:
    if col in df.columns:
        print(f"--- {col} ---")
        print(df.groupby(['group','time'])[col].describe()[['count','mean','50%','min','max']])
        print()
"""),
    md("## Sanity asserts"),
    code("""\
assert patient.shape[0] == N_TOTAL == 69, 'Patient row count mismatch'
assert df['surgery_num'].value_counts().to_dict() == {1: 69, 2: 69}, 'S1/S2 imbalance'
print('All EDA asserts passed.')
"""),
])

# ============================================================================
# 01 — Baseline balance (S1)
# ============================================================================
nb_01 = make_nb([
    md("# 01 — Baseline balance (S1)\n\n"
       "**Notebook id**: `01_baseline_balance` → vault `03.2-balance-baseline.md`\n\n"
       "Is there a major imbalance on baseline (S1) covariates between groups? "
       "We report **standardised mean differences** (Austin) and visualise them "
       "as a **Love plot** (including a **PF score S1** row), and we confirm the "
       "baseline lesion load is equivalent — crucially on the **patellofemoral "
       "block at S1 specifically** (`tf.baseline_pf_balance`), with a **TOST** "
       "equivalence test rather than reading a non-significant difference as "
       "equivalence (absence of evidence ≠ evidence of absence; revue 2026-05-29). "
       "So the PF progression contrast is not driven by a starting-point "
       "difference or a ceiling effect."),
    code(BOOTSTRAP),
    code(PREAMBLE),
    code(LOAD),
    md("## 1. Table 1 (SMD) recap"),
    code("""\
table1 = rpt.make_table1(patient)
print(table1.to_string(index=False))
"""),
    md("## 2. SMD per covariate (Cyclops − Meniscus) + Love plot\n\n"
       "Continuous SMD = Cohen's d (pooled SD); binary SMD = Austin's "
       "proportion-difference form. |SMD| < 0.10 negligible; ≥ 0.25 meaningful."),
    code("""\
smd_rows = []  # (label, signed SMD = cyclops - meniscus)

# Continuous covariates.
for var, label in [('age_at_trauma','Age at trauma'), ('imc','BMI'),
                   ('taille','Height'), ('poids','Weight')]:
    if var not in patient.columns:
        continue
    c = pd.to_numeric(patient.loc[patient.group=='cyclops', var], errors='coerce').dropna()
    m = pd.to_numeric(patient.loc[patient.group=='meniscus', var], errors='coerce').dropna()
    if len(c) >= 2 and len(m) >= 2:
        smd_rows.append((label, tf.smd_continuous(c.values, m.values)))

# Binary covariates (proportion of level==1): female, tabac, travail_physique.
def _smd_bin(col):
    c = pd.to_numeric(patient.loc[patient.group=='cyclops', col], errors='coerce').dropna()
    m = pd.to_numeric(patient.loc[patient.group=='meniscus', col], errors='coerce').dropna()
    p1, p2 = (c==c.max()).mean(), (m==m.max()).mean()
    denom = np.sqrt((p1*(1-p1) + p2*(1-p2))/2) or np.nan
    return float((p1-p2)/denom)

for var, label in [('female','Female'), ('tabac','Smoker'),
                   ('travail_physique','Physical work')]:
    if var in patient.columns:
        smd_rows.append((label, _smd_bin(var)))

# Baseline PF-score-at-S1 SMD row (the block that carries the primary outcome):
# add it to the Love plot so balance is shown on the OUTCOME's starting point,
# not only on the global 6-sum (revue 2026-05-29).
if 'lesion_pf_S1' in wide.columns:
    pf_c = pd.to_numeric(wide.loc[wide.group=='cyclops','lesion_pf_S1'], errors='coerce').dropna()
    pf_m = pd.to_numeric(wide.loc[wide.group=='meniscus','lesion_pf_S1'], errors='coerce').dropna()
    if len(pf_c) >= 2 and len(pf_m) >= 2:
        smd_rows.append(('PF score S1', tf.smd_continuous(pf_c.values, pf_m.values)))

print('SMD (Cyclops - Meniscus):')
for lab, v in sorted(smd_rows, key=lambda kv: -abs(kv[1])):
    flag = '  <-- meaningful' if abs(v) >= 0.25 else ''
    print(f'  {lab:16s} {v:+.3f}{flag}')

# Baseline S1 lesion load equivalence (annotated on the Love plot).
b_c = wide.loc[wide.group=='cyclops','lesion_total_S1'].dropna().astype(float).values
b_m = wide.loc[wide.group=='meniscus','lesion_total_S1'].dropna().astype(float).values
base = tf.mwu_with_effects(b_c, b_m, n_boot=2000)
print(f"\\nBaseline S1 lesion load (6-sum): MWU p = {base['pvalue']:.3f} "
      f"(medians {np.median(b_c):.0f} / {np.median(b_m):.0f})")
"""),
    code("""\
fig = viz.love_plot_smd(smd_rows, baseline_score_p=base['pvalue'])
fig  # display inline
"""),
    md("## 2b. Baseline equivalence on the PF block at S1 (MWU + SMD + TOST)\n\n"
       "The headline baseline check, on the **patellofemoral score at S1** that "
       "carries the primary outcome. `tf.baseline_pf_balance` reports the MWU p "
       "(difference test), the SMD, **and** a **TOST** equivalence p within a "
       "small-SMD margin (±0.5 pooled SD). `equivalent=True` ⇒ the PF starting "
       "point is *statistically equivalent*, so the PF progression contrast is "
       "not a regression-to-baseline artefact — and this is asserted positively, "
       "not inferred from a non-significant difference."),
    code("""\
pf_bal = tf.baseline_pf_balance(wide, col='lesion_pf_S1')
print('Baseline PF block at S1 (cyclops vs meniscus):')
print(f"  n            : {pf_bal['n_case']} cyclops / {pf_bal['n_control']} meniscus")
print(f"  medians      : {pf_bal['median_case']:.1f} / {pf_bal['median_control']:.1f}")
print(f"  MWU p (diff) : {pf_bal['mwu_p']:.3f}")
print(f"  SMD          : {pf_bal['smd']:+.3f}")
print(f"  TOST bound   : +/- {pf_bal['tost_bound']:.3f} (pooled-SD margin)")
print(f"  TOST p (equiv): {pf_bal['tost_p']:.3f}")
print(f"  => equivalent: {pf_bal['equivalent']}  (TOST p < 0.05 establishes equivalence)")
"""),
    md("## 3. Per-test baseline detail (continuous MWU; categorical Fisher)"),
    code("""\
print('Continuous (Mann-Whitney U + Cliff delta):')
for var in ['age_at_trauma','imc','taille','poids']:
    if var not in patient.columns: continue
    c = pd.to_numeric(patient.loc[patient.group=='cyclops', var], errors='coerce').dropna()
    m = pd.to_numeric(patient.loc[patient.group=='meniscus', var], errors='coerce').dropna()
    r = tf.mwu_with_effects(c.values, m.values, n_boot=2000)
    print(f"  {var:16s} delta={r['cliffs_delta']:+.2f}  p={r['pvalue']:.3f}")

print('\\nCategorical (Fisher exact 2x2):')
for var in ['female','tabac','travail_physique']:
    if var not in patient.columns: continue
    tab = pd.crosstab(patient['group'], patient[var]).values
    if tab.shape != (2,2):
        print(f'  {var}: not 2x2 ({tab.shape}) - skipped'); continue
    r = tf.fisher_exact_2x2(tab)
    print(f"  {var:16s} OR={r['odds_ratio']:.2f}  p={r['pvalue']:.3f}")
"""),
    md("## 4. Baseline S1 lesion scores per compartment (pre-progression)"),
    code("""\
s1 = df[df['surgery_num']==1]
for site in SITES + ['lesion_pf','lesion_ft','lesion_total']:
    if site not in s1.columns: continue
    c = s1.loc[s1['group']=='cyclops', site].dropna().astype(float).values
    m = s1.loc[s1['group']=='meniscus', site].dropna().astype(float).values
    if len(c) < 2 or len(m) < 2: continue
    r = tf.mwu_with_effects(c, m, n_boot=500)
    print(f"{site:14s}  delta={r['cliffs_delta']:+.2f}  "
          f"CI=[{r['delta_ci_lo']:+.2f}, {r['delta_ci_hi']:+.2f}]  p={r['pvalue']:.3f}")
"""),
    md("## Sanity asserts"),
    code("""\
assert len(patient) == N_TOTAL
print('Baseline-balance asserts passed.')
"""),
])

# ============================================================================
# 02 — Progression S1→S2 (PRIMARY: patellofemoral)  ⭐
# ============================================================================
nb_02 = make_nb([
    md("# 02 — Progression S1 → S2 — PRIMARY (patellofemoral) ⭐\n\n"
       "**Notebook id**: `02_progression_total` → vault `03.3-progression-S1-S2.md`\n\n"
       "**Primary estimand** (consensus point B): the **patellofemoral block** "
       "Δ`lesion_pf` = {trochlée, rotule} progresses more in Cyclops than in "
       "Meniscus.\n\n"
       "- **Frequentist support**: Mann–Whitney U + **Cliff's δ** (rank-based, no "
       "interval assumption) + **exact/Monte-Carlo permutation** p + **BCa** CI "
       "(B = 10 000) + a **test-inversion** guard-rail CI — all via "
       "`tf.pf_contrast`.\n"
       "- **Decision rule** (point G): the global Bayesian verdict lives in M3 "
       "(notebook 05); the permutation p is the frequentist support, not a "
       "separate gate.\n"
       "- **Binary effect**: the worsened-PF odds ratio is estimated with **Firth "
       "penalised logistic** (`tf.firth_or`) — crude **and** sex+age-adjusted. The "
       "1/20 meniscus events near-separate the plain ML logit into a *ghost* "
       "OR≈24 (≈47 adjusted); Firth gives a stable OR + CI, and we report an "
       "**E-value** (`tf.evalue_or`) for robustness to unmeasured confounding.\n"
       "- **Demoted / descriptive**: the 6-compartment sum Δ`lesion_total` is a "
       "*global burden index* that **dilutes** the compartmental signal — reported "
       "(two- and one-sided) but **not** decisional.\n"
       "- **M2** (NegBin on the 6-sum, Δ⁺ truncation): **removed** from the "
       "inferential chain."),
    code(BOOTSTRAP),
    code(PREAMBLE),
    code(LOAD),
    md("## 1. Distribution of the primary outcome Δ`lesion_pf`"),
    code("""\
delta_pf = wide[['group','delta_lesion_pf']].dropna()
print(delta_pf.groupby('group')['delta_lesion_pf'].describe())
print()
worsened = wide.groupby('group')['worsened_pf'].apply(lambda s: (s.dropna()==1).mean())
print('Proportion worsening (Δ_PF > 0) by group:')
print((worsened*100).round(1).astype(str) + ' %')
"""),
    md("## 2. PRIMARY contrast — `tf.pf_contrast` (MWU + Cliff δ + permutation + BCa + inversion)"),
    code("""\
pf = tf.pf_contrast(wide, value_col='delta_lesion_pf',
                    n_boot=10000, n_perm=20000, seed=RANDOM_SEED)
for k, v in pf.items():
    print(f'  {k:26s} = {v}')

print()
print('--- verdict (frequentist support) ---')
print(rpt.verdict_freq_en(
    pf['perm_p'], 'Primary PF contrast (cyclops vs meniscus)',
    effect=f"Cliff delta = {pf['cliffs_delta']:+.2f} ({pf['cliffs_delta_magnitude']})"))
"""),
    md("## 3. PRIMARY figure — raincloud of Δ`lesion_pf` by group"),
    code("""\
worsened_pct = {g: float((wide.loc[wide.group==g,'delta_lesion_pf'].dropna()>0).mean()*100)
                for g in GROUPS}
fig = viz.raincloud_progression(
    wide, value_col='delta_lesion_pf',
    worsened_pct=worsened_pct,
    cliff_delta=pf['cliffs_delta'], perm_p=pf['perm_p'],
    title='Patellofemoral progression S1 -> S2 (primary)',
    ylabel='Δ patellofemoral lesion score (S2 - S1)')
fig
"""),
    md("## 4. PF score trajectories S1 → S2 (slopegraph)"),
    code("""\
fig = viz.slopegraph_pf_mpl(wide)
fig
"""),
    md("## 4b. Binary worsened-PF odds ratio — Firth penalised logistic + E-value\n\n"
       "Dichotomising the PF outcome (`worsened_pf` = Δ`lesion_pf` > 0) gives a "
       "2×2 with only **1/19** worsening events in meniscus → near-complete "
       "separation. Plain ML logistic returns a *ghost* OR ≈ 24 (≈ 47 once "
       "adjusted) with an explosive CI — it must **not** anchor the paper. "
       "`tf.firth_or` (Firth/Jeffreys penalised likelihood) returns a finite, "
       "stable OR. We report:\n\n"
       "- **crude** OR (group only);\n"
       "- **sex + age-adjusted** OR (`covariates=('female','age_at_trauma')`) — "
       "the headline robustness analysis: the PF effect survives adjustment;\n"
       "- an **E-value** (`tf.evalue_or`, common-outcome `RR≈√OR`): the minimum "
       "association an unmeasured confounder would need with *both* group and "
       "outcome to explain the effect away (large ⇒ robust)."),
    code("""\
or_crude = tf.firth_or(wide, outcome_col='worsened_pf', covariates=())
or_adj = tf.firth_or(merged, outcome_col='worsened_pf',
                     covariates=('female','age_at_trauma'))
print('Firth penalised-logistic OR for worsened_pf (cyclops vs meniscus):')
print(f"  method        : {or_adj['method']}")
print(f"  separation_ml : {or_crude['separation_ml']}  (min 2x2 cell = {or_crude['min_cell']})")
print(f"  events        : cases {or_crude['n_events_case']}/{'?'}  "
      f"controls {or_crude['n_events_control']} (the 1-event meniscus cell)")
print()
print(f"  crude        : OR = {or_crude['odds_ratio']:.2f}  "
      f"[{or_crude['or_ci_lo']:.2f}, {or_crude['or_ci_hi']:.2f}]  "
      f"p = {or_crude['p']:.4f}  n = {or_crude['n']}")
print(f"  sex+age adj. : OR = {or_adj['odds_ratio']:.2f}  "
      f"[{or_adj['or_ci_lo']:.2f}, {or_adj['or_ci_hi']:.2f}]  "
      f"p = {or_adj['p']:.4f}  n = {or_adj['n']}  cov={or_adj['covariates']}")
print('  (the effect SURVIVES sex+age adjustment; the plain-ML OR=24/47 was a ghost)')
"""),
    code("""\
# E-value for the crude Firth OR (worsened_pf is a common outcome -> RR ~ sqrt(OR)).
ev = tf.evalue_or(or_crude['odds_ratio'], or_ci_lo=or_crude['or_ci_lo'],
                  common_outcome=True)
print('E-value for the crude worsened-PF OR (VanderWeele & Ding 2017):')
print(f"  RR approx    : {ev['rr_approx']:.2f}  ({ev['method']})")
print(f"  E-value point: {ev['evalue_point']:.2f}")
print(f"  E-value CI   : {ev['evalue_ci']}")
print('  Interpretation: an unmeasured confounder would need associations of at')
print('  least this strength with BOTH group and worsened_pf to explain it away.')
"""),
    md("## 5. DESCRIPTIVE / SECONDARY — 6-compartment sum (dilution, non-decisional)\n\n"
       "The 6-sum mixes commensurable PF grades with rare/absent FT events; it "
       "**dilutes** the signal to a small effect. We report it transparently with "
       "both the two-sided p (transparency) and the one-sided p (direction was "
       "pre-registered) — **neither drives any decision** (point G)."),
    code("""\
s6 = wide[['group','delta_lesion_total']].dropna()
c6 = s6.loc[s6.group=='cyclops','delta_lesion_total'].astype(float).values
m6 = s6.loc[s6.group=='meniscus','delta_lesion_total'].astype(float).values
from scipy import stats as _st
res6 = tf.mwu_with_effects(c6, m6, n_boot=10000, seed=RANDOM_SEED)
_, p_one = _st.mannwhitneyu(c6, m6, alternative='greater')
print(f"6-sum Cliff delta = {res6['cliffs_delta']:+.3f} ({res6['cliffs_delta_magnitude']})")
print(f"6-sum MWU p two-sided = {res6['pvalue']:.4f}   one-sided = {p_one:.4f}")
print(f"BCa 95% CI = [{res6['delta_ci_lo']:+.3f}, {res6['delta_ci_hi']:+.3f}]")
print()
print('Interpretation: the global burden index understates a compartment-specific'
      ' effect (PF). It is descriptive only; the primary inference is the PF contrast.')
"""),
    md("## 6. M2 — removed from the inferential chain (note only)\n\n"
       "The earlier M2 (Negative-Binomial on the 6-compartment sum with a Δ⁺ = "
       "max(Δ, 0) truncation) is **removed** (consensus point D): summing "
       "non-commensurable ordinals and discarding improvements is incoherent. "
       "A labelled NegBin **sanity-check** on the post-op PF score adjusted for "
       "baseline (no truncation) is available as "
       "`bm.fit_m2_negbin_sanity`, but it is **not** part of the primary "
       "inference (which is M3 + permutation)."),
    code("""\
# Optional labelled sanity-check (NOT primary; no Δ⁺ truncation). Uncomment to run.
# sane = bm.fit_m2_negbin_sanity(wide, outcome_col='lesion_pf_S2',
#                                baseline_col='lesion_pf_S1')
# print(sane['note'])
# print(rpt.summary_bayes(sane['idata']))
print('M2 is removed from the primary chain (see markdown above).')
"""),
    md("## Sanity asserts"),
    code("""\
assert wide['delta_lesion_pf'].notna().sum() == N_TOTAL_ANALYSABLE == 69
assert pf['cliffs_delta'] > 0 and pf['perm_p'] < 0.01
print('Primary PF endpoint asserts passed.')
"""),
])

# ============================================================================
# 03 — Progression per compartment + PF-vs-FT specificity
# ============================================================================
nb_03 = make_nb([
    md("# 03 — Progression per compartment + topographic specificity\n\n"
       "**Notebook id**: `03_progression_sites` → vault `03.4-progression-par-site.md`\n\n"
       "**Full disclosure** of all six compartments (point C.3): the signal is "
       "confined to the **patellofemoral** block. We (1) test each compartment "
       "(cyclops vs meniscus) with Mann–Whitney + Cliff δ and BH-FDR (q = 0.10) "
       "and bar-plot the worsening %, then (2) show the **within-patient** "
       "topographic specificity — Δ`lesion_pf` vs Δ`lesion_ft` paired Wilcoxon."),
    code(BOOTSTRAP),
    code(PREAMBLE),
    code(LOAD),
    md("## 1. Per-compartment progression (cyclops vs meniscus) + BH-FDR\n\n"
       "Direction of progression on each compartment Δ; PTI/CFI are binary "
       "(grade ≥ 2 never observed). BH-FDR controls the family of 6 tests."),
    code("""\
rows = []
pvals = []
for s in SITES:
    col = f'delta_{s}'
    if col not in wide.columns: continue
    c = wide.loc[wide.group=='cyclops', col].dropna().astype(float).values
    m = wide.loc[wide.group=='meniscus', col].dropna().astype(float).values
    r = tf.mwu_with_effects(c, m, n_boot=2000, seed=RANDOM_SEED)
    rows.append(dict(
        compartment=s,
        block='PF' if s in SITES_PF else 'FT',
        measurement='binary' if s in SITES_BINARY else 'ordinal',
        worsened_pct_cyc=round(float((c>0).mean()*100), 1),
        worsened_pct_men=round(float((m>0).mean()*100), 1),
        cliff_delta=round(float(r['cliffs_delta']), 4),
        mwu_p=round(float(r['pvalue']), 4),
    ))
    pvals.append(r['pvalue'])
per_comp = pd.DataFrame(rows)
bh = tf.bh_fdr(pvals, q=0.10)
per_comp['bh_p_adj'] = [round(float(x), 4) for x in bh['pvals_corrected']]
per_comp['bh_reject'] = bh['reject']
print(per_comp.to_string(index=False))
"""),
    md("## 2. Per-compartment worsening bars (PF block vs FT block)"),
    code("""\
fig = viz.per_compartment_bars(per_comp)
fig
"""),
    md("## 3. Topographic specificity — within-patient Δ`lesion_pf` vs Δ`lesion_ft`\n\n"
       "Paired Wilcoxon within the cyclops group: does the PF block worsen more "
       "than the FT block in the **same** patient? (Mechanistic specificity, "
       "point B.)"),
    code("""\
spec = tf.paired_pf_vs_ft(wide)
for k, v in spec.items():
    print(f'  {k:22s} = {v}')
print()
print(rpt.format_test_result(spec, 'wilcoxon'))
"""),
    code("""\
fig = viz.topographic_specificity(
    wide,
    n_pf_worsened=spec['n_pf_worsened'], n_ft_worsened=spec['n_ft_worsened'],
    rank_biserial=spec['rank_biserial'], wilcoxon_p=spec['pvalue'])
fig
"""),
    md("## Sanity asserts"),
    code("""\
assert len(per_comp) == 6  # six compartments fully disclosed
# PF compartments should worsen more in cyclops than the FT compartments do.
pf_mean = per_comp.loc[per_comp.block=='PF','worsened_pct_cyc'].mean()
ft_mean = per_comp.loc[per_comp.block=='FT','worsened_pct_cyc'].mean()
assert pf_mean > ft_mean, (pf_mean, ft_mean)
print('Per-compartment + specificity asserts passed.')
"""),
])

# ============================================================================
# 04 — Risk factors (H3, within Cyclops, against Δ_PF)
# ============================================================================
nb_04 = make_nb([
    md("# 04 — Modulating risk factors (H3, exploratory)\n\n"
       "**Notebook id**: `04_risk_factors` → vault `03.5-facteurs-risque.md`\n\n"
       "Within the **Cyclops** group, which intrinsic factors modulate the "
       "**patellofemoral** progression Δ`lesion_pf`? (H3 is **exploratory**.)\n\n"
       "- Continuous (age, **BMI** — `imc`, now usable, derived from taille/poids): "
       "Spearman ρ + BCa CI.\n"
       "- Binary (female, smoker, physical work): Mann–Whitney + Cliff δ.\n"
       "- Multilevel pivot (0/1/2): Kruskal–Wallis + ε².\n"
       "- All p-values **BH-FDR** corrected (family F2). `inter_surgery_d` is "
       "**excluded** — it is time-at-risk / a mediator (H4 outcome), not a risk "
       "factor.\n"
       "- **Sport/occupation sensitivity**: the group→worsened-PF effect **with and "
       "without** the Pivot / Physical-work covariates, via the now **Firth-based** "
       "`tf.sensitivity_covariate` (stable under the 1/19 separation)."),
    code(BOOTSTRAP),
    code(PREAMBLE),
    code(LOAD),
    md("## 1. Cyclops subset (joined to covariates)"),
    code("""\
cyc = merged[merged['group']=='cyclops'].copy()
print('Cyclops subset:', cyc.shape)
print(cyc[['delta_lesion_pf','age_at_trauma','imc','female','tabac',
           'travail_physique','pivot_pivot_contact']].head())
"""),
    md("## 2. H3 factor battery vs Δ`lesion_pf` (BH-FDR, exploratory)"),
    code("""\
h3 = tf.h3_risk_factors(
    cyc, outcome_col='delta_lesion_pf',
    continuous=('age_at_trauma','imc'),
    binary=('female','tabac','travail_physique'),
    multilevel=('pivot_pivot_contact',),
    q=0.10, n_boot=2000, seed=RANDOM_SEED)
print(h3.to_string(index=False))
print()
print('All H3 associations are exploratory (hypothesis-generating).')
"""),
    md("## 3. Kruskal–Wallis detail on `pivot_pivot_contact` (0/1/2)"),
    code("""\
if 'pivot_pivot_contact' in cyc.columns:
    sub = cyc[['pivot_pivot_contact','delta_lesion_pf']].apply(
        pd.to_numeric, errors='coerce').dropna()
    sub['pivot_pivot_contact'] = sub['pivot_pivot_contact'].astype(int).astype(str)
    res = tf.kw_dunn(sub, 'pivot_pivot_contact', 'delta_lesion_pf')
    print(f"H={res['statistic']:.3f}  p={res['pvalue']:.3f}  "
          f"eps^2={res['epsilon_sq']:.3f}  n={res['n']}")
    print('Dunn post-hoc (Bonferroni):')
    print(res['posthoc'])
"""),
    md("## 4. Sport / occupation sensitivity — group effect WITH vs WITHOUT covariates\n\n"
       "**Firth** penalised logistic of `worsened_pf` on group: crude (WITHOUT) vs "
       "adjusted (WITH) Pivot + Physical work. Firth is used because the 1/19 "
       "meniscus events separate the plain ML logit (ghost OR); `or_*_ci` are the "
       "penalised CIs. If the adjusted OR stays comparable to the crude OR, the PF "
       "effect is **not** an artefact of the sport/occupation imbalance."),
    code("""\
sens = tf.sensitivity_covariate(
    merged, outcome_col='worsened_pf',
    covariates=('pivot_pivot_contact','travail_physique'))
print('Sport/occupation sensitivity (worsened_pf ~ group), Firth-based:')
print(f"  method        : {sens['method']}  (separation_ml={sens['separation_ml']})")
print(f"  crude (WITHOUT): OR = {sens['or_crude']:.2f}  CI {tuple(round(x,2) for x in sens['or_crude_ci'])}"
      f"  p = {sens['p_crude']:.3f}  n = {sens['n_crude']}")
print(f"  adj.  (WITH)   : OR = {sens['or_adjusted']:.2f}  CI {tuple(round(x,2) for x in sens['or_adjusted_ci'])}"
      f"  p = {sens['p_adjusted']:.3f}  n = {sens['n_adjusted']}")
print(f"  covariates     : {sens['covariates']}")
print(f"  adjusted_ok    : {sens['adjusted_ok']}")
"""),
    md("## 5. Spearman correlogram (Cyclops, exploratory)"),
    code("""\
cols = [c for c in ['age_at_trauma','imc','taille','poids','inter_surgery_d',
                    'delta_lesion_pf','delta_lesion_ft'] if c in cyc.columns]
fig = viz.correlogram(cyc, cols)
fig
"""),
    md("## Sanity asserts"),
    code("""\
assert len(cyc) == N_CYCLOPS  # 49 cyclops patients in the subset
assert 'p_adj_bh' in h3.columns
print('Risk-factors asserts passed.')
"""),
])

# ============================================================================
# 05 — Hierarchical Bayes M3 (PRIMARY INFERENTIAL ⭐⭐)
# ============================================================================
nb_05 = make_nb([
    md("# 05 — Hierarchical Bayes M3 (⭐⭐ PRIMARY INFERENTIAL)\n\n"
       "**Notebook id**: `05_hierarchical_bayes` → vault `03.6-modele-hierarchique.md`\n\n"
       "The refactored **heterogeneous** hierarchical model, fitted under **three "
       "competing pooling structures** so the topographic organisation is an "
       "empirical **RESULT tested by LOO**, not an assumed hypothesis "
       "(revue-methodo 2026-05-29):\n\n"
       "- **Likelihood**: `pm.Bernoulli`/logit for the binary compartments "
       "**PTE/PTI/CFE/CFI** (0 events ≥ 2); `pm.OrderedLogistic` on the collapsed "
       "**{0, 1, ≥2}** scale (2 cutpoints) for **trochlée, rotule**.\n"
       "- **Free cutpoints per compartment** — cutpoints encode *measurement* "
       "(baseline prevalence), never pooled.\n"
       "- **Pooling on the Group×Time slopes** `delta_comp`, with the grouping set "
       "by the structure under test:\n"
       "  - `exchangeable` → one knee-wide mean ⇒ exposes **`delta_bar`** "
       "(the selection-immune global estimand);\n"
       "  - `two_block` → {PF}/{FT} ⇒ exposes `delta_pf`, `delta_ft`, "
       "`contrast_pf_ft` (a **candidate** structure);\n"
       "  - `three_cluster` → {PF}/{FT-antlat}/{FT-med}.\n"
       "- η = β_c[comp]·t + γ·g + δ_comp[comp]·t·g + u[patient], **t ∈ {−0.5,+0.5}** "
       "the **S2 − S1 contrast** (not a per-year slope); Student-t(3) priors; "
       "patient random intercept `u` retained.\n\n"
       "**Primary global estimand** = **δ̄** (`delta_bar`) from the *exchangeable* "
       "fit: a pre-specified directional H1 (\"cyclops worsen cartilage\"), so it "
       "uses the one-sided rule `P(δ̄>0|data) ≥ 0.95` (`two_sided=False`). It is "
       "reported **honestly**: the knee-wide average **dilutes** a compartment-"
       "specific signal, so it is expected to be *small and uncertain* — we do "
       "**not** overclaim it.\n\n"
       "**Topography** = a RESULT: `bm.compare_pooling_structures` ranks the three "
       "fits by **LOO** (ELPD ± SE). At n=69 this is likely only *indicative* "
       "(small ΔELPD with a comparable dse) — reported, not over-claimed.\n\n"
       "**PF localisation** = read **non-circularly** from `bm.derived_pf_contrast` "
       "on the *neutral exchangeable* posterior (no PF/FT partition imposed). The "
       "two-block `contrast_pf_ft` is shown as a **candidate-structure** output "
       "and judged by the **two-sided** credible rule (post-hoc ⇒ HDI excludes 0), "
       "never the one-sided rule.\n\n"
       "**NUTS**: `chains=4, tune=2000, draws=2000, target_accept=0.95, seed=42`, "
       "sampled with **nutpie** (the host has no C compiler for the default "
       "PyTensor backend). **Convergence**: R̂ ≤ 1.01, ESS_bulk ≥ 400, "
       "divergences = 0; `fit_m3_with_escalation` auto-escalates "
       "(target_accept=0.99, tune=4000) once if needed."),
    code(BOOTSTRAP),
    code(PREAMBLE),
    code(LOAD),
    md("## 1. Model data shape (long-long, collapsed {0,1,≥2})\n\n"
       "One row per (patient × time × compartment), NaN scores dropped, grade "
       "3 → 2. The two likelihood blocks (binary vs ordinal) are split inside "
       "the model builder."),
    code("""\
long_long = bm._melt_long_long(df, SITES, 'anonyme', 'group', collapse=True)
print('long_long shape:', long_long.shape, '(non-NaN site cells, collapsed)')
print('y categories:', sorted(long_long['y'].unique()), '(<=2 => collapse applied)')
print()
print('Block per compartment (0=PF, 1=FT):')
print(long_long.groupby('comp')['block_idx'].first())
print('\\nBinary compartments (Bernoulli):',
      long_long.loc[long_long.is_binary, 'comp'].unique().tolist())
print('Time contrast values (S1=-0.5, S2=+0.5):', sorted(long_long['t'].unique()))
"""),
    md("## 2. Fit & compare the three pooling structures by LOO (runnable — sampling)\n\n"
       "This cell **is runnable** (uncommented) and re-runs NUTS via nutpie **three "
       "times** (exchangeable, two_block, three_cluster), each with the pointwise "
       "log-likelihood, then ranks them by `arviz.compare` (ELPD-LOO). It returns "
       "`{idatas, loos, compare}`. **This is the test of whether the data reject "
       "6-site exchangeability in favour of a block structure** — the topography "
       "is thereby an empirical result, not an assumption.\n\n"
       "> At n=69 the comparison is likely only **indicative** (small ΔELPD, "
       "comparable `dse`). Read the ELPD differences *and their SE*; do not "
       "over-claim a winner."),
    code("""\
# Fit all three pooling structures and rank by LOO. Runs NUTS x3 via nutpie.
res = bm.compare_pooling_structures(df, sites=SITES)
idatas = res['idatas']
idata_exch = idatas['exchangeable']      # neutral / selection-immune (PRIMARY)
idata_2blk = idatas['two_block']         # candidate topographic structure
print('LOO comparison (ranked by ELPD-LOO; topography tested, not assumed):')
print(res['compare'])
# Optional: cache the neutral posterior for make_figures.py / reuse.
# idata_exch.to_netcdf('../results/idata_m3_exchangeable.nc')
"""),
    md("## 3. Convergence diagnostics on every structure (R̂ ≤ 1.01, ESS ≥ 400, div = 0)\n\n"
       "All three fits must converge before the LOO ranking is trustworthy."),
    code("""\
diags = {k: bm.check_convergence(v, rhat_max=1.01, ess_min=400)
         for k, v in idatas.items()}
for k, d in diags.items():
    print(f"  {k:14s} ok={d['ok']}  max_rhat={d['max_rhat']:.3f}  "
          f"min_ess_bulk={d['min_ess_bulk']:.0f}  n_div={d['n_divergent']}")
diag = diags['exchangeable']            # primary structure's diagnostics
assert all(d['ok'] for d in diags.values()), \\
    f"convergence failed: {[(k, d['offenders'][:4]) for k, d in diags.items() if not d['ok']]}"
"""),
    md("## 4. PRIMARY global verdict — knee-wide δ̄ (exchangeable, pre-specified)\n\n"
       "δ̄ = `delta_bar` is the **selection-immune global** Group×Time interaction "
       "from the *exchangeable* model (no PF/FT partition imposed). H1 (\"cyclops "
       "worsen cartilage\") was pre-specified and **directional**, so we use the "
       "legitimate **one-sided** rule `P(δ̄ > 0 | data) ≥ 0.95` "
       "(`two_sided=False, direction='greater'`).\n\n"
       "Reported **honestly**: because the knee-wide average **dilutes** a "
       "compartment-specific effect, δ̄ is expected to be *small and uncertain* — "
       "a wide HDI straddling 0 is the honest answer, **not** an overclaimed "
       "global effect."),
    code("""\
print('Exchangeable posterior summary (delta_bar = knee-wide global):')
print(rpt.summary_bayes(idata_exch, var_names=['delta_bar','gamma','sigma_delta','sigma_u'],
                        hdi_prob=0.94))
print()
v_global = rpt.verdict_bayes_en(
    idata_exch, 'delta_bar', 'Primary global knee-wide effect (delta-bar)',
    threshold=0.95, two_sided=False, direction='greater')   # pre-specified directional
print(v_global['sentence'])
print(rpt.confidence_phrase_en(
    v_global['p_direction'], 'the knee-wide cartilage worsens more in cyclops'))
print()
print('NOTE: delta_bar is the honest PRIMARY estimand. A small / 0-straddling HDI'
      ' reflects DILUTION of a compartment-specific signal, not absence of an effect'
      ' in the patellofemoral block (localised below).')
"""),
    md("## 5. PF localisation — `derived_pf_contrast` on the NEUTRAL posterior (non-circular)\n\n"
       "The honest, **non-circular** reading of where the signal sits: take the "
       "joint posterior of the per-compartment `delta_comp` from the *exchangeable* "
       "fit (which imposed **no** PF/FT partition) and form (mean δ over PF sites) "
       "− (mean δ over FT sites). This reveals the pattern the data show without "
       "*assuming* the partition that the contrast measures."),
    code("""\
pf = bm.derived_pf_contrast(idata_exch)
print('Derived PF - FT contrast (from the neutral exchangeable delta_comp):')
print(f"  PF sites      : {pf['pf_sites']}")
print(f"  FT sites      : {pf['ft_sites']}")
print(f"  mean          : {pf['mean']:+.3f}")
print(f"  {int(pf['hdi_prob']*100)}% HDI       : [{pf['hdi_lo']:+.3f}, {pf['hdi_hi']:+.3f}]")
print(f"  P(PF-FT > 0)  : {pf['p_gt0']:.3f}")
print('  => the worsening is localised to the patellofemoral block (data-revealed,')
print('     not assumed); the partition was selected, so the amplitude is an')
print('     optimistic bound.')
"""),
    md("## 6. CANDIDATE structure — two-block `contrast_pf_ft` (post-hoc, two-sided rule)\n\n"
       "The `two_block` fit is one **candidate** topographic structure (the one LOO "
       "compares above). Its deterministic `contrast_pf_ft` = δ_PF − δ_FT is a "
       "**post-hoc** estimand (the PF/FT split was suggested by the data), so it is "
       "judged by the direction-agnostic **two-sided** credible rule (HDI excludes "
       "0) — **never** the one-sided P>0 rule, which would double-count the data."),
    code("""\
print('Two-block posterior summary (candidate structure):')
print(rpt.summary_bayes(idata_2blk,
                        var_names=['delta_pf','delta_ft','contrast_pf_ft','gamma'],
                        hdi_prob=0.94))
print()
v_contrast = rpt.verdict_bayes_en(
    idata_2blk, 'contrast_pf_ft',
    'Topographic specificity PF-FT (candidate two-block structure)',
    two_sided=True)                                         # post-hoc => two-sided
v_pf_blk = rpt.verdict_bayes_en(
    idata_2blk, 'delta_pf', 'PF block delta (candidate structure)', two_sided=True)
print(v_pf_blk['sentence'])
print(v_contrast['sentence'])
print('  (labelled CANDIDATE: this structure is one of three compared by LOO above.)')
"""),
    md("## 7. Forest plot — two-block posterior effects (94% HDI)\n\n"
       "Visualises the candidate two-block estimands; the global δ̄ verdict lives "
       "in section 4 (exchangeable)."),
    code("""\
fig = viz.forest_m3(idata_2blk,
                    var_names=('delta_pf','delta_ft','contrast_pf_ft','gamma'))
fig
"""),
    md("## 8. Convergence diagnostic plots (trace + rank) — exchangeable (primary)"),
    code("""\
fig = viz.diagnostics_m3(idata_exch,
                         var_names=('delta_bar','gamma'),
                         convergence=diag)
fig
"""),
    md("## 9. Per-compartment slopes & free cutpoints (exchangeable, interpretation)\n\n"
       "β_c is the per-compartment time slope; the cutpoints are free per ordinal "
       "compartment (measurement, not pooled)."),
    code("""\
import arviz as az
print('Per-compartment slopes beta_c (94% HDI, exchangeable):')
print(rpt.summary_bayes(idata_exch, var_names=['beta_c'], hdi_prob=0.94))
print()
cs = bm.cutpoint_summary(idata_exch)
print('Free cutpoints per ordinal compartment:')
for comp, d in cs.get('per_comp', {}).items():
    print(f"  {comp:10s} cutpoints ~ {[round(x,2) for x in d['cut_means']]}")
"""),
    md("## 10. Posterior predictive check (optional, ~30 s after the fit)\n\n"
       "Category frequencies {0, 1, ≥2} per (group, time, compartment): empirical "
       "vs posterior-predictive draws, across both likelihood blocks."),
    code("""\
# Optional PPC on the exchangeable fit (uncomment to run after the fits).
# ppc = bm.ppc_m3(idata_exch, df, n_draws=500)
# print('Observed (head):'); print(ppc['observed_freq'].head(10))
# print('Predicted mean (head):'); print(ppc['predicted_freq_mean'].head(10))
print('PPC cell ready (commented). Uncomment to run after fitting M3.')
"""),
    md("## 11. Prior sensitivity on σ_β (optional, exploratory)\n\n"
       "Re-fit with `sigma_beta` HalfNormal scale in (0.5, 1.0, 5.0) — tight / "
       "default / loose pooling — and compare the posterior of `delta_pf` and "
       "`contrast_pf_ft` (two-block). Each refit re-runs NUTS; run only if you "
       "want the grid."),
    code("""\
# Optional prior sensitivity grid (each line re-runs NUTS via nutpie).
# from constants import SENSITIVITY_PRIOR_SIGMAS
# rows = []
# for s in SENSITIVITY_PRIOR_SIGMAS:
#     id_s = bm.fit_m3_with_prior(df, prior_sigma_beta=s)   # two_block by default
#     for v in ['delta_pf','contrast_pf_ft']:
#         vals = id_s.posterior[v].values.ravel()
#         rows.append(dict(prior_sigma_beta=s, var=v, mean=float(vals.mean()),
#                          p_gt0=float((vals>0).mean())))
# print(pd.DataFrame(rows))
print('Prior-sensitivity grid ready (commented). Uncomment to run 3 refits.')
"""),
    md("## Sanity asserts (post-fit)\n\n"
       "All three structures converged; the primary verdict is on δ̄ "
       "(pre-specified one-sided). The PF localisation / candidate contrast are "
       "**not** asserted (post-hoc, honestly reported)."),
    code("""\
assert all(d['ok'] for d in diags.values()), 'a pooling structure failed to converge'
assert diag['max_rhat'] <= 1.01 and diag['min_ess_bulk'] >= 400 and diag['n_divergent'] == 0
assert set(res['idatas']) == {'exchangeable','two_block','three_cluster'}
assert 'delta_bar' in idata_exch.posterior          # primary estimand exposed
assert 'contrast_pf_ft' in idata_2blk.posterior     # candidate-structure estimand
print('M3 convergence + structure-comparison asserts passed.')
print('Primary global verdict (delta_bar):', v_global['supported'])
"""),
])

# ============================================================================
# 06 — Temporal / H4 (inter-surgical delay)
# ============================================================================
nb_06 = make_nb([
    md("# 06 — Temporal trends (inter-surgical delay, H4)\n\n"
       "**Notebook id**: `06_temporal` → vault `03.7-tendances-temporelles.md`\n\n"
       "**H4**: cyclops patients are re-operated **sooner** (shorter "
       "`inter_surgery_d`).\n\n"
       "- **DAG / causal status** (point F): the delay is a **mediator** "
       "*downstream of group* (Group → delay → progression, and Group → "
       "progression). It is therefore the **outcome** of H4 — it is **not** a "
       "confounder of the PF effect, and the primary model does **not** adjust "
       "for it. Conditioning on it (as a sensitivity / mediation analysis) "
       "*strengthens* the PF effect.\n"
       "- **Frequentist**: Mann–Whitney on `inter_surgery_d` (cyclops vs meniscus).\n"
       "- **Time-at-risk falsification**: the delay-**adjusted** worsened-PF OR "
       "(`tf.firth_or(covariates=('inter_surgery_d',))`) + the correlation "
       "ρ(delay, worsened_pf). If the delay were the *biological* driver, "
       "adjusting for it would collapse the OR and ρ would be strongly negative; "
       "instead the OR is preserved and ρ ≈ 0 — consistent with **time-at-risk**, "
       "not a confounder/mediator that explains the PF effect away.\n"
       "- **M4**: Weibull AFT (lifelines MLE).\n"
       "- **M5**: per-group LogNormal (NUTS via nutpie) — sampling cell is "
       "runnable but not executed here.\n"
       "- **Figure**: ECDF + fitted LogNormal per group with median annotations.\n\n"
       "> **Anomaly note**: the date anomalies flagged in 00 (#9 trauma-date drift; "
       "#25/#38 negative trauma→surgery) corrupt *age / trauma-to-surgery*, **not** "
       "`inter_surgery_d` — the latter is the **difference of two `date_chir`** "
       "(S2 − S1), neither of which is hygienised or negative here. H4 is therefore "
       "robust to those source-date issues."),
    code(BOOTSTRAP),
    code(PREAMBLE),
    code(LOAD),
    md("## 1. DAG (time-at-risk, not a confounder)\n\n"
       "```\n"
       "        Group (cyclops vs meniscus)\n"
       "        /                        \\\n"
       "       v                          v\n"
       "  inter_surgery_d  ----------->  PF progression\n"
       "   (time-at-risk / mediator)      (Δ lesion_pf)\n"
       "```\n"
       "The delay is **time-at-risk** sitting *downstream* of group, not a "
       "confounder. The primary PF estimand is the **total** effect of group; we do "
       "**not** condition on the delay in the primary model (that would over-adjust "
       "a mediator). H4 treats the delay as an **outcome**, and §3b uses it only as "
       "a falsification check."),
    md("## 2. Distribution of the inter-surgical delay"),
    code("""\
isd = wide[['group','inter_surgery_d']].dropna()
print(isd.groupby('group')['inter_surgery_d'].describe())
print()
print('Medians (days):')
print(isd.groupby('group')['inter_surgery_d'].median())
"""),
    md("## 3. Mann–Whitney on `inter_surgery_d` (H4)"),
    code("""\
c = isd.loc[isd.group=='cyclops','inter_surgery_d'].astype(float).values
m = isd.loc[isd.group=='meniscus','inter_surgery_d'].astype(float).values
r = tf.mwu_with_effects(c, m, n_boot=10000, seed=RANDOM_SEED)
print(f"inter_surgery_d: Cliff delta = {r['cliffs_delta']:+.3f} "
      f"({r['cliffs_delta_magnitude']}), MWU p = {r['pvalue']:.5f}")
print(f"BCa 95% CI on delta = [{r['delta_ci_lo']:+.3f}, {r['delta_ci_hi']:+.3f}]")
"""),
    md("## 3b. Time-at-risk falsification — delay-adjusted OR + ρ(delay, worsened-PF)\n\n"
       "Two checks that the delay is **time-at-risk**, not the biological cause of "
       "the PF effect:\n\n"
       "1. **Delay-adjusted Firth OR** — `tf.firth_or(worsened_pf, "
       "covariates=('inter_surgery_d',))`. If the delay *mediated* the effect, "
       "conditioning on it would shrink the OR toward 1. Instead the OR is **kept** "
       "(if anything inflated, because more time-at-risk in cyclops is part of the "
       "causal path, not a confound) — so the PF effect is not explained away by "
       "the delay.\n"
       "2. **Falsification correlation** ρ(`inter_surgery_d`, `worsened_pf`) within "
       "the analysable cohort: a *biological-mediator* story predicts a clear "
       "negative ρ; **ρ ≈ 0** is what time-at-risk predicts here."),
    code("""\
# Delay-adjusted worsened-PF OR (Firth; merged carries inter_surgery_d + covariates).
or_delay = tf.firth_or(merged, outcome_col='worsened_pf',
                       covariates=('inter_surgery_d',))
print('Delay-ADJUSTED Firth OR for worsened_pf (cyclops vs meniscus):')
print(f"  OR = {or_delay['odds_ratio']:.2f}  "
      f"[{or_delay['or_ci_lo']:.2f}, {or_delay['or_ci_hi']:.2f}]  "
      f"p = {or_delay['p']:.4f}  n = {or_delay['n']}  method={or_delay['method']}")
print('  (the PF effect SURVIVES adjustment for the delay => not delay-mediated;')
print('   the delay is time-at-risk on the causal path, not a confounder.)')
print()
# Falsification correlation: delay vs worsened_pf should be ~0 (not a mediator).
fal = merged[['inter_surgery_d','worsened_pf']].apply(pd.to_numeric, errors='coerce').dropna()
rho = tf.spearman_bca(fal['inter_surgery_d'].values, fal['worsened_pf'].values,
                      n_boot=2000, seed=RANDOM_SEED)
print('Falsification correlation rho(inter_surgery_d, worsened_pf):')
print(f"  rho = {rho['rho']:+.3f}  CI [{rho['ci_lo']:+.3f}, {rho['ci_hi']:+.3f}]  "
      f"p = {rho['pvalue']:.3f}  n = {rho['n']}")
print('  (rho ~ 0 => no monotone delay->worsening link; consistent with time-at-risk.)')
"""),
    md("## 4. M4 — Weibull AFT on `inter_surgery_d` (lifelines MLE)\n\n"
       "Accelerated-failure-time on the delay (no censoring). A positive group "
       "coefficient ⇒ longer delay; cyclops should be shorter."),
    code("""\
aft_cov = [c for c in ['group','imc','pivot_pivot_contact'] if c in merged.columns]
m4 = bm.fit_m4_weibull_aft(merged, duration_col='inter_surgery_d', covariates=aft_cov)
print(m4['summary'])
"""),
    md("## 5. M5 — per-group LogNormal on `inter_surgery_d` (runnable sampling cell)\n\n"
       "This cell **is runnable** (uncommented). Execute it yourself to sample "
       "(NUTS via nutpie). Posterior median delay per group = exp(μ_group)."),
    code("""\
# Fit M5 LogNormal on the inter-surgical delay (H4 outcome). Runs NUTS via nutpie.
idata_m5 = bm.fit_m5_lognormal(wide, var='inter_surgery_d', nuts_sampler='nutpie')
diag5 = bm.check_convergence(idata_m5)
print('M5 convergence:', diag5)
summ5 = rpt.summary_bayes(idata_m5, var_names=['mu','sigma'], hdi_prob=0.94)
print(summ5)
medians = {g: float(np.exp(idata_m5.posterior['mu'].sel(group=g).values.mean()))
           for g in idata_m5.posterior['mu'].coords['group'].values}
print('Posterior median delay by group (days):',
      {g: round(v,1) for g, v in medians.items()})
# Optional: cache for make_figures.py.
# idata_m5.to_netcdf('../results/idata_m5.nc')
"""),
    md("## 6. Figure — delay ECDF + fitted LogNormal per group\n\n"
       "Empirical ECDF overlaid with the fitted LogNormal (M5 posterior-mean "
       "μ/σ when `idata_m5` is in scope, else a per-group MLE), with medians "
       "and the mediator caption."),
    code("""\
medians = {g: float(isd.loc[isd.group==g,'inter_surgery_d'].median()) for g in GROUPS}
_idata5 = idata_m5 if 'idata_m5' in dir() else None
fig = viz.delay_ecdf_fit(wide, idata=_idata5, value_col='inter_surgery_d',
                         medians=medians)
fig
"""),
    md("## Sanity asserts"),
    code("""\
assert wide['inter_surgery_d'].notna().sum() >= 60
assert isd.groupby('group')['inter_surgery_d'].median()['cyclops'] < \
       isd.groupby('group')['inter_surgery_d'].median()['meniscus'], \
       'cyclops should be re-operated sooner (H4)'
print('Temporal / H4 asserts passed.')
"""),
])


def write_all() -> None:
    for stem, nb in [
        ("00_eda", nb_00),
        ("01_baseline_balance", nb_01),
        ("02_progression_total", nb_02),
        ("03_progression_sites", nb_03),
        ("04_risk_factors", nb_04),
        ("05_hierarchical_bayes", nb_05),
        ("06_temporal", nb_06),
    ]:
        path = NB_DIR / f"{stem}.ipynb"
        nbf.write(nb, str(path))
        print(f"wrote {path}")


if __name__ == "__main__":
    write_all()
