"""Data hygiene, derived variables, and wide/long pivots.

Pipeline (typical use)::

    import loaders, preprocessing as pp
    df = loaders.load_combined()
    df = pp.apply_date_hygiene(df)
    df = pp.add_surgery_num(df)
    df = pp.add_derived(df)
    wide = pp.to_wide(df)
    patient = pp.to_patient(df)

All functions are pure (return new frames, never mutate in place).
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from constants import SITES, SITES_PF, SITES_FT


# Conversion: 1 year = 365.25 days (Julian year, contract default)
DAYS_PER_YEAR: float = 365.25


# --- Hygiene ---------------------------------------------------------------

def apply_date_hygiene(
    df: pd.DataFrame,
    date_cols: Iterable[str] = ("date_de_naissance", "date_du_trauma"),
    id_col: str = "anonyme",
    group_col: str = "group",
) -> pd.DataFrame:
    """Resolve 1-day intra-patient drift on static dates by taking the min.

    Some patients have ``date_de_naissance`` and ``date_du_trauma`` that
    differ by 1 day between their S1 and S2 rows (manual data-entry rounding).
    Per ``02-Methods/02.2-variables-coding.md`` §4.1, we replace by the
    minimum within the patient.

    .. note::
       **Composite patient key.** The ``Anonyme`` id is *reused* across the
       two sheets — 19 ids (1–19) appear in **both** meniscus and cyclops
       (revue-methodo-2026-05 §2.6, consensus point I.1). Grouping by
       ``id_col`` alone therefore merges distinct patients and corrupts the
       birth/trauma dates of the shared ids. We group by the composite key
       ``[group_col, id_col]`` so each physical patient is treated separately.
       ``date_chir`` is **not** hygienised (it is the row-level discriminator
       that defines S1 vs S2).

    Parameters
    ----------
    df : pd.DataFrame
        Long frame with one row per (patient × surgery).
    date_cols : iterable of str
        Date columns to hygienise.
    id_col : str
        Patient identifier column.
    group_col : str
        Group/sheet column completing the composite patient key.

    Returns
    -------
    pd.DataFrame
        Copy with hygienised dates.
    """
    out = df.copy()
    by = [group_col, id_col] if group_col in out.columns else [id_col]
    for col in date_cols:
        if col in out.columns:
            # Surface genuine inconsistencies BEFORE collapsing to the min:
            # min() silently hides a real divergence (revue-methodo 2026-05-29:
            # patient #9 had trauma dates 2 months apart, not 1-day rounding).
            span = out.groupby(by)[col].transform(
                lambda s: (s.dropna().max() - s.dropna().min()).days
                if s.notna().sum() >= 2 else 0
            )
            drift = span > 1
            if drift.any():
                bad = out.loc[drift, by].drop_duplicates()
                warnings.warn(
                    f"apply_date_hygiene: {bad.shape[0]} patient(s) with "
                    f"intra-patient drift > 1 day on {col!r} (resolved by min; "
                    f"verify the source): {bad.to_dict('records')}",
                    stacklevel=2,
                )
            out[col] = out.groupby(by)[col].transform("min")
    return out


# --- Date-anomaly detection (revue-methodo 2026-05-29) ---------------------

def detect_date_anomalies(
    df: pd.DataFrame,
    id_col: str = "anonyme",
    group_col: str = "group",
    drift_tol_days: int = 1,
) -> pd.DataFrame:
    """Tidy frame of date inconsistencies that corrupt age/delay (NOT the PF Δ).

    Detects, per composite patient ``(group, anonyme)``:

    * **static-date drift** — ``date_de_naissance`` / ``date_du_trauma`` differing
      by more than ``drift_tol_days`` between the S1 and S2 rows (e.g. patient #9
      had trauma dates ~2 months apart). :func:`apply_date_hygiene` resolves these
      by ``min`` but the divergence flags a source-data issue to fix by hand.
      **Call this on the RAW frame (before** :func:`apply_date_hygiene` **)** so
      the drift is still visible; negative delays are detectable before or after.
    * **negative trauma_to_surgery_d** — surgery dated *before* the trauma
      (impossible; e.g. #38 = −488 d), almost always a placeholder/typo date.

    None of these touch the cartilage scores or the patellofemoral Δ (which
    contain no dates); they only pollute ``age_at_trauma``,
    ``trauma_to_surgery_d``, baseline balance, H3 and H4. ``run_all`` exports
    this as ``results/data_anomalies.csv`` for manual source correction, and the
    unique ``(group, anonyme)`` keys form the exclusion list for the
    delay-analysis sensitivity (with/without).

    Returns
    -------
    pd.DataFrame
        Columns ``[group, anonyme, kind, detail]`` (empty frame if none found).
    """
    by = [group_col, id_col] if group_col in df.columns else [id_col]
    rows: list[dict] = []

    def _key_dict(key) -> dict:
        return dict(zip(by, key if isinstance(key, tuple) else (key,)))

    for col in ("date_de_naissance", "date_du_trauma"):
        if col in df.columns:
            for key, sub in df.groupby(by):
                vals = sub[col].dropna()
                if len(vals) >= 2:
                    span = (vals.max() - vals.min()).days
                    if span > drift_tol_days:
                        rec = _key_dict(key)
                        rec["kind"] = f"{col}_drift"
                        rec["detail"] = (
                            f"{span} d between rows "
                            f"({vals.min().date()} to {vals.max().date()})"
                        )
                        rows.append(rec)

    # Recompute the raw delay from dates (NOT the derived column, which
    # add_derived has already coerced to NaN for negatives) so the anomaly stays
    # detectable post-pipeline.
    if {"date_chir", "date_du_trauma"}.issubset(df.columns):
        raw = (df["date_chir"] - df["date_du_trauma"]).dt.days
        neg = df.loc[raw.notna() & (raw < 0)].assign(_tts=raw)
        for key, sub in neg.groupby(by):
            rec = _key_dict(key)
            rec["kind"] = "negative_trauma_to_surgery_d"
            rec["detail"] = f"min {int(sub['_tts'].min())} d (surgery before trauma)"
            rows.append(rec)
    elif "trauma_to_surgery_d" in df.columns:
        neg = df[df["trauma_to_surgery_d"].notna() & (df["trauma_to_surgery_d"] < 0)]
        for key, sub in neg.groupby(by):
            rec = _key_dict(key)
            rec["kind"] = "negative_trauma_to_surgery_d"
            rec["detail"] = (
                f"min {int(sub['trauma_to_surgery_d'].min())} d (surgery before trauma)"
            )
            rows.append(rec)

    return pd.DataFrame(rows, columns=by + ["kind", "detail"])


# --- Scale collapse {0, 1, ≥2} ---------------------------------------------

def collapse_scores(
    df: pd.DataFrame,
    sites: Iterable[str] = tuple(SITES),
) -> pd.DataFrame:
    """Collapse the native 0–3 ordinal scale to {0, 1, ≥2} (grade 3 → 2).

    Consensus point A (revue-methodo-2026-05 §3.A): the modelling scale is the
    collapsed {0, 1, ≥2} grade, where the upper category groups Outerbridge 2
    (deep cartilage loss) and 3 (exposed bone) into a single "advanced lesion"
    pole. The collapse is information-neutral on the patellofemoral signal
    (§2.2: PF Cliff δ 0.533 → 0.532; rotule 0.505 → 0.484) and makes the upper
    cutpoints identifiable. The native 0–3 scale is retained elsewhere for the
    descriptive raw-frequency table and sensitivity analyses.

    Parameters
    ----------
    df : pd.DataFrame
        Long frame containing the site columns.
    sites : iterable of str
        Compartment columns to collapse (default: package ``SITES``).

    Returns
    -------
    pd.DataFrame
        Copy with the listed site columns mapped 3 → 2 (other values
        unchanged, NaN preserved). Columns stay nullable ``Int64``.
    """
    out = df.copy()
    for col in list(sites):
        if col in out.columns:
            s = out[col].astype("Int64")
            # ``s == 3`` is pd.NA on missing entries; ``Series.mask`` would
            # treat an NA condition as "mask" and overwrite real NaNs with 2.
            # Fill the condition with False so only genuine 3s are remapped.
            cond = (s == 3).fillna(False)
            out[col] = s.mask(cond, 2).astype("Int64")
    return out


# --- Surgery numbering ----------------------------------------------------

def add_surgery_num(
    df: pd.DataFrame,
    id_col: str = "anonyme",
    group_col: str = "group",
    date_col: str = "date_chir",
) -> pd.DataFrame:
    """Rank surgeries 1, 2 within each (group × patient) by ``date_chir``.

    The rank is computed by ascending ``date_chir`` so the first chronological
    intervention is ``surgery_num=1`` (S1). A categorical ``time`` column with
    labels ``S1`` / ``S2`` is also added for plotting.

    Parameters
    ----------
    df : pd.DataFrame
    id_col, group_col, date_col : str

    Returns
    -------
    pd.DataFrame
        Copy with new columns ``surgery_num`` (Int64) and ``time`` (str).
    """
    out = df.copy()
    out["surgery_num"] = (
        out.groupby([group_col, id_col])[date_col]
        .rank(method="first", ascending=True)
        .astype("Int64")
    )
    out["time"] = out["surgery_num"].map({1: "S1", 2: "S2"})
    return out


# --- Derived variables -----------------------------------------------------

def add_derived(
    df: pd.DataFrame,
    sites: Iterable[str] = tuple(SITES),
    id_col: str = "anonyme",
    group_col: str = "group",
) -> pd.DataFrame:
    """Add lesion_total, lesion_pf/ft, female, age_at_*, trauma_to_surgery_d, …

    Will also enforce ``Int64`` on site columns and ensure ``surgery_num``
    exists (calling :func:`add_surgery_num` if missing).

    Parameters
    ----------
    df : pd.DataFrame
    sites : iterable of str
        Compartment column names (default: package SITES constant).
    id_col, group_col : str

    Returns
    -------
    pd.DataFrame
        Copy with derived columns appended.
    """
    out = df.copy()

    # Cast sites to nullable Int64
    sites = list(sites)
    for col in sites:
        if col in out.columns:
            out[col] = out[col].astype("Int64")

    # --- lesion_total: STRICT (analysis convention) -----------------------
    # Consensus point I.5: ``lesion_total`` is NaN if ANY site is missing
    # (``min_count = len(sites)``). This is the version used in analysis so a
    # row with a partially-observed cartilage grid never produces a spuriously
    # low total. The permissive "sum of whatever is present" variant is kept
    # as ``lesion_total_permissive`` for descriptive coverage diagnostics, and
    # ``lesion_total_strict`` is retained as an explicit alias.
    present = [s for s in sites if s in out.columns]
    out["lesion_total"] = (
        out[present].sum(axis=1, min_count=len(present)).astype("Int64")
    )
    out["lesion_total_strict"] = out["lesion_total"]
    out["lesion_total_permissive"] = (
        out[present].sum(axis=1, min_count=1).astype("Int64")
    )
    out["n_sites_observed"] = out[present].notna().sum(axis=1).astype("Int64")
    out["lesion_any"] = (out["lesion_total"] > 0).astype("Int64")

    # --- Topographic block sums (consensus point B) -----------------------
    # PF = {trochlée, rotule}; FT = {pte, pti, cfe, cfi}. Both strict
    # (NaN if any constituent site is missing) so the block contrast is on a
    # complete sub-grid.
    pf_present = [s for s in SITES_PF if s in out.columns]
    ft_present = [s for s in SITES_FT if s in out.columns]
    if pf_present:
        out["lesion_pf"] = (
            out[pf_present].sum(axis=1, min_count=len(pf_present)).astype("Int64")
        )
    if ft_present:
        out["lesion_ft"] = (
            out[ft_present].sum(axis=1, min_count=len(ft_present)).astype("Int64")
        )

    # --- Sex indicator (consensus point I.4) ------------------------------
    # Sexe ∈ {1, 2}. Anthropometric proof (revue-methodo §"FAITS DONNÉES"):
    # Sexe=1 → mean 1.79 m / 84 kg, Sexe=2 → 1.66 m / 65 kg → 1 = Homme (male),
    # 2 = Femme (female). ``female`` = 1 iff Sexe == 2.
    if "sexe" in out.columns:
        out["female"] = (
            pd.to_numeric(out["sexe"], errors="coerce") == 2
        ).astype("Int64")

    # --- IMC (BMI): derive from taille/poids when not recorded ------------
    # ``imc`` is empty in this cohort (0/138) yet height and weight are complete
    # → compute BMI = poids / taille² so the clinician-flagged adiposity
    # confounder is usable in H3 / M4 / sensitivity (revue 2026-05-29). ``taille``
    # is in metres here; fall back to cm→m for any value that looks like cm.
    if {"taille", "poids"}.issubset(out.columns):
        _imc = pd.to_numeric(out["imc"], errors="coerce") if "imc" in out.columns else None
        if _imc is None or _imc.notna().sum() == 0:
            _taille = pd.to_numeric(out["taille"], errors="coerce")
            _poids = pd.to_numeric(out["poids"], errors="coerce")
            _taille_m = _taille.where(_taille < 3.0, _taille / 100.0)
            out["imc"] = (_poids / (_taille_m ** 2)).round(2)

    # surgery_num (idempotent — adds if missing)
    if "surgery_num" not in out.columns:
        out = add_surgery_num(out, id_col=id_col, group_col=group_col)

    # Age and delay variables (days then years)
    if {"date_de_naissance", "date_du_trauma"}.issubset(out.columns):
        days_trauma = (out["date_du_trauma"] - out["date_de_naissance"]).dt.days
        out["age_at_trauma"] = days_trauma / DAYS_PER_YEAR
    if {"date_de_naissance", "date_chir"}.issubset(out.columns):
        days_surg = (out["date_chir"] - out["date_de_naissance"]).dt.days
        out["age_at_surgery"] = days_surg / DAYS_PER_YEAR
    if {"date_chir", "date_du_trauma"}.issubset(out.columns):
        tts = (out["date_chir"] - out["date_du_trauma"]).dt.days
        n_neg = int((tts < 0).sum())
        if n_neg:
            # Surgery dated before the trauma is impossible (e.g. #38 = −488 d):
            # a placeholder/typo date. Coerce to NaN so it drops out of delay /
            # H4 / baseline analyses listwise (flagged by detect_date_anomalies).
            warnings.warn(
                f"add_derived: {n_neg} negative trauma_to_surgery_d "
                f"(surgery before trauma, data anomaly) coerced to NaN.",
                stacklevel=2,
            )
            tts = tts.where(tts >= 0)
        out["trauma_to_surgery_d"] = tts.astype("Int64")

    return out


# --- Pivots ---------------------------------------------------------------

def to_wide(
    df_long: pd.DataFrame,
    sites: Iterable[str] = tuple(SITES),
    id_col: str = "anonyme",
    group_col: str = "group",
) -> pd.DataFrame:
    """Pivot long → wide: one row per (group × patient), one Δ per site.

    Computes per-patient deltas ``delta_<site> = site(S2) - site(S1)`` and
    ``delta_total = lesion_total(S2) - lesion_total(S1)``, plus the
    inter-surgery delay ``inter_surgery_d = date_chir(S2) - date_chir(S1)``.

    Parameters
    ----------
    df_long : pd.DataFrame
        Must contain ``surgery_num`` (run :func:`add_derived` first).
    sites : iterable of str
    id_col, group_col : str

    Returns
    -------
    pd.DataFrame
        Expected shape ``(69, N)``.
    """
    if "surgery_num" not in df_long.columns:
        raise ValueError("df_long must contain 'surgery_num' — call add_derived first.")

    sites = list(sites)
    block_cols = ["lesion_pf", "lesion_ft"]
    total_cols = ["lesion_total", "lesion_total_strict", "lesion_total_permissive"]
    values = sites + block_cols + total_cols + [
        "lesion_any", "n_sites_observed", "date_chir", "trauma_to_surgery_d",
    ]
    values = [v for v in values if v in df_long.columns]

    wide = df_long.pivot_table(
        index=[group_col, id_col],
        columns="surgery_num",
        values=values,
        aggfunc="first",
        observed=False,
    )
    # Flatten MultiIndex columns: ('trochlée', 1) -> 'trochlée_S1'
    wide.columns = [f"{a}_S{int(b)}" for a, b in wide.columns]
    wide = wide.reset_index()

    # Compute Δ per site, per block (PF/FT) and Δtotal variants.
    delta_targets = sites + block_cols + total_cols
    for col_base in delta_targets:
        s1 = f"{col_base}_S1"
        s2 = f"{col_base}_S2"
        if s1 in wide.columns and s2 in wide.columns:
            wide[f"delta_{col_base}"] = wide[s2] - wide[s1]

    # Inter-surgery delay (days)
    if "date_chir_S1" in wide.columns and "date_chir_S2" in wide.columns:
        wide["inter_surgery_d"] = (
            wide["date_chir_S2"] - wide["date_chir_S1"]
        ).dt.days.astype("Int64")

    # Worsening flags (per site, per block, and global). worsened_X = Δ > 0.
    if "delta_lesion_total" in wide.columns:
        wide["worsened_any"] = (wide["delta_lesion_total"] > 0).astype("Int64")
    for col_base in sites + block_cols:
        col = f"delta_{col_base}"
        if col in wide.columns:
            flag = f"worsened_{col_base.replace('lesion_', '')}"
            wide[flag] = (wide[col] > 0).astype("Int64")

    return wide


def to_patient(
    df_long: pd.DataFrame,
    static_cols: Iterable[str] = (
        "sexe",
        "female",
        "taille",
        "poids",
        "imc",
        "pivot_pivot_contact",
        "travail_physique",
        "tabac",
        "age_at_trauma",
    ),
    id_col: str = "anonyme",
    group_col: str = "group",
) -> pd.DataFrame:
    """Collapse long frame to one row per patient with static attributes.

    Static attributes are reduced via ``first`` (after :func:`apply_date_hygiene`
    these are constant within patient anyway).

    Parameters
    ----------
    df_long : pd.DataFrame
    static_cols : iterable of str
    id_col, group_col : str

    Returns
    -------
    pd.DataFrame
        Expected shape ``(69, N)``.
    """
    cols = [c for c in static_cols if c in df_long.columns]
    out = (
        df_long.groupby([group_col, id_col], as_index=False)[cols]
        .first()
    )
    return out
