# lino_stats

Dual-population (Cyclops n=49 vs Ménisque n=20 contrôle, 69 patients) bayésien + non-paramétrique statistical analysis of a paired knee-surgery cohort.

## Pipeline flow

```
data/data_paper_cyclops_stats.xlsx
        │
        ▼
loaders.load_combined()          (138 rows × 20 cols)
        │
        ▼
preprocessing.apply_date_hygiene │
preprocessing.add_derived        │  long  (138 × ~25)
        │
        ├─► to_wide()    → wide    (69 × ~37)   ← Δ per site, Δtotal, delays
        └─► to_patient() → patient (69 × ~10)   ← static covariates
                │
                ▼
        tests_freq (MWU/Wilcoxon/Fisher/KW/Spearman + Cliff's δ + BCa + BH-FDR)
        bayes_models (M1 Beta-Bin / M2 NegBin / M3 hier-ordinal / M4 AFT / M5 LogN)
        reporting (Table 1 / summary_bayes ArviZ / markdown export)
        viz (slopegraph, Sankey, heatmap, dumbbell, raincloud, trace, PPC, ...)
```

## Quick start

```python
# modules live flat in src/ — run with src/ on the path (PYTHONPATH=src)
import loaders, preprocessing as pp, tests_freq as tf

df = loaders.load_combined()  # (138, 20) long
df = pp.apply_date_hygiene(df)
df = pp.add_derived(df)
wide = pp.to_wide(df)  # (69, ...) one row / patient

# H1 primary: progression Cyclops vs Méniscus on Δlesion_total
a = wide.loc[wide["group"] == "meniscus", "delta_lesion_total"].astype(float).values
b = wide.loc[wide["group"] == "cyclops", "delta_lesion_total"].astype(float).values
res = tf.mwu_with_effects(a, b, n_boot=5000, ci=0.95)
print(
    f"U={res['statistic']:.1f}  p={res['pvalue']:.3f}  "
    f"Cliff's δ={res['cliffs_delta']:+.3f} "
    f"[{res['delta_ci_lo']:+.3f}, {res['delta_ci_hi']:+.3f}]"
)
```

Expected output: `U=390.0  p=0.156  Cliff's δ=-0.204 [-0.446, +0.074]` (Δlesion_total, méniscus vs cyclops — diluted by the 4 inert FT compartments; the localized PF signal is in §02/§03).

## Install

Python **3.11+** requis (testé sur 3.13).

```bash
pip install -e .
pytest tests/ -v          # 48 tests, doit afficher 48 passed
```

## Run notebooks

```bash
jupyter notebook notebooks/
```

Lire `notebooks/` dans l'ordre numéroté. Chaque notebook = un chapitre Results du wiki Obsidian.

Re-génération des fichiers `.ipynb` depuis le script source :

```bash
python notebooks/_make_notebooks.py
```

Exécution headless d'un notebook :

```bash
jupyter nbconvert --to notebook --execute notebooks/00_eda.ipynb --output 00_eda.ipynb
```

## Structure

- `src/`  modules (layout plat, sur le `PYTHONPATH`) :
  - `loaders.py`  lecture Excel 2 onglets + normalisation
  - `preprocessing.py`  data hygiene, dérivés, wide/long
  - `viz.py`  seaborn/plotly/altair (imports paresseux)
  - `tests_freq.py`  MWU/Wilcoxon/Fisher/KW/Spearman + Cliff's δ + BCa CI + BH-FDR
  - `bayes_models.py`  PyMC M1 (Beta-Bin) / M2 (NegBin) / **M3 (hierarchical ordinal logit) ** / M4 (Weibull AFT) / M5 (LogNormal)
  - `reporting.py`  Table 1, ArviZ summaries, export markdown
- `notebooks/`  7 notebooks chapitres + `_make_notebooks.py` (générateur)
- `figures/`  sorties (`<nb_id>_<slug>.{png,html}`)
- `tests/test_pipeline.py`  sanity asserts
- `data/data_paper_cyclops_stats.xlsx` + `data/flexum.xlsx`  données source (gitignored, lues par `loaders`)
- `stats.ipynb`  **ARCHIVÉ phase 1 exploratoire** (voir `stats.ipynb.HISTORICAL.md`)

## Mapping notebooks → vault

| Notebook                        | Note vault                        |
| ------------------------------- | --------------------------------- |
| `00_eda.ipynb`                | `03.1-EDA-descriptive.md`       |
| `01_baseline_balance.ipynb`   | `03.2-balance-baseline.md`      |
| `02_progression_total.ipynb`  | `03.3-progression-S1-S2.md`     |
| `03_progression_sites.ipynb`  | `03.4-progression-par-site.md`  |
| `04_risk_factors.ipynb`       | `03.5-facteurs-risque.md`       |
| `05_hierarchical_bayes.ipynb` | `03.6-modele-hierarchique.md`   |
| `06_temporal.ipynb`           | `03.7-tendances-temporelles.md` |

## Cellules Bayes

Les cellules `pm.sample(...)` (M2, M3, M5) sont **commentées** dans les notebooks `02`, `05`, `06` pour permettre un run rapide. Décommenter pour lancer (M3 ≈ 5–15 min, 4 chaînes × 2000 draws). Vérifier `bm.check_convergence(idata)`.

## Hypothèse principale

**H1** : la cohorte Cyclops montre une progression lésionnelle entre S1 et S2 statistiquement supérieure à celle de Ménisque.

Voir wiki `02-Methods/02.3-strategie-stats.md` pour stratégie complète.

## Diagnostics convergence (verrouillés)

Tous modèles bayésiens : `r_hat ≤ 1.01`, `ess_bulk ≥ 400`, divergences=0.

## Reproductibilité

`RANDOM_SEED=42` partout. Notebooks idempotents en kernel fresh.

### Fresh-kernel reproducibility check

Pour vérifier qu'un notebook produit les mêmes résultats à chaque exécution :

```bash
# Run any notebook fresh and overwrite in place
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
    notebooks/00_eda.ipynb --output 00_eda.ipynb

# Compare hash of two consecutive runs
python -c "import hashlib; print(hashlib.md5(open('notebooks/00_eda.ipynb','rb').read()).hexdigest())"
```

(Note : cellules `pm.sample(...)` commentées par défaut. L'idempotence stricte exige
`random_seed=RANDOM_SEED` dans tout appel `pm.sample()` / `bootstrap()`  ce qui est
respecté dans `lino_stats.bayes_models` et `lino_stats.tests_freq`.)

## Troubleshooting

- **matplotlib `_c_internal_utils.cp39-win_amd64.pyd` missing on Python 3.13** : env utilisateur
  possède une copie matplotlib 3.9.4 compilée pour Python 3.9. Voir décision
  `99-Project-Log/decisions.md` (2026-05-27, python). `viz.py` utilise des imports
  paresseux pour rester importable malgré ce backend cassé ; les helpers Plotly
  fonctionnent. Fix recommandé : `pip install --force-reinstall matplotlib` dans
  l'env Python 3.13.
- **`ArviZ.from_dict` TypeError** : ArviZ ≥ 1.0 a remplacé `from_dict(posterior=...)`
  par `from_dict({"posterior": {...}})`. `tests/test_pipeline.py::test_summary_bayes_api`
  gère le fallback automatiquement.
- **NUTS divergences > 0** : voir table d'escalation dans
  `02-Methods/02.5-modeles-bayesiens.md` §7  bascule sur
  `lino_stats.NUTS_KWARGS_ESCALATED` (target_accept=0.99, tune=4000).
