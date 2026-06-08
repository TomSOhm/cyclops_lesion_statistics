"""Shared fixtures for the lino_stats test suite.

Centralises the data-load chain (Excel → date-hygiene → derived → wide/patient)
that ~12 tests previously repeated inline. The workbook is read **once** per
session (``df_long``); the cheap pivots (``wide``, ``patient``, ``merged``) are
re-derived per test so no test can mutate another's frame.

Tests that exercise a *raw* load stage on purpose (e.g. ``test_load_*``,
``test_detect_date_anomalies_flags_known``) keep their own inline load  they
must not go through the hygiene/derive chain these fixtures apply.
"""

from __future__ import annotations

import warnings

import pytest

import loaders
import preprocessing as pp

# Covariate columns joined onto the wide frame for adjusted/sensitivity models
# (mirrors run_all.py); filtered to those actually present.
_COV_COLS = [
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


@pytest.fixture(scope="session")
def df_long():
    """Combined long frame after date-hygiene + derived columns (read once)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # data-anomaly UserWarnings are expected
        df = loaders.load_combined()
        df = pp.apply_date_hygiene(df)
        df = pp.add_derived(df)
    return df


@pytest.fixture
def wide(df_long):
    """One row per (group × patient) with per-block Δ columns (fresh per test)."""
    return pp.to_wide(df_long)


@pytest.fixture
def patient(df_long):
    """Patient-level frame with covariates (fresh per test)."""
    return pp.to_patient(df_long)


@pytest.fixture
def merged(wide, patient):
    """Wide frame joined with patient covariates  for adjusted/sensitivity models."""
    cols = [c for c in _COV_COLS if c in patient.columns]
    return wide.merge(patient[cols], on=["group", "anonyme"], how="left")
