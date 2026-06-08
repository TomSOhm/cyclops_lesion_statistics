"""Load and normalise the dual-cohort Excel dataset.

The source file has two sheets (``Ménisque`` and ``Cyclop``) with identical
columns but slightly different dtypes (Méniscus lesions are int64 while
Cyclops contains a few NaN and is therefore float64). Loaders normalise
column names (strip / lowercase / underscore-separated, slashes mapped to
underscores) and append the ``group`` column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

DATA_PATH: Path = Path(__file__).resolve().parents[1] / "data" / "data_paper_cyclops_stats.xlsx"

# Flexum (pre-S2 extension deficit, in degrees)  a separate workbook with the
# same two-sheet (Ménisque / Cyclop) layout but a single measurement column.
# Negative values = loss of extension (a fixed-flexion / flexum, the mechanical
# driver of patellofemoral overload in cyclops syndrome). Kept out of git
# (patient data, see .gitignore); loaded locally for the pipeline.
FLEXUM_PATH: Path = Path(__file__).resolve().parents[1] / "data" / "flexum.xlsx"


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace, lowercase, replace ' / ' and ' ' by '_'.

    Parameters
    ----------
    df : pd.DataFrame
        Raw frame read from one of the two sheets.

    Returns
    -------
    pd.DataFrame
        Same data with column labels normalised. Examples::

            "Date de naissance" -> "date_de_naissance"
            "Pivot / pivot contact" -> "pivot_pivot_contact"
            "Trochlée" -> "trochlée"
            "Sexe " -> "sexe"
    """
    df = df.rename(columns=lambda x: x.strip())
    df = df.rename(
        columns=lambda x: (
            x.replace(" / ", "_").replace("/", "_").replace(" ", "_").lower()
        )
    )
    return df


def load_meniscus(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the Ménisque (meniscus group) sheet and append ``group='meniscus'``.

    Parameters
    ----------
    path : Path, default DATA_PATH
        Excel workbook path.

    Returns
    -------
    pd.DataFrame
        Expected shape ``(38, 20)`` (19 patients × 2 surgeries).
    """
    df = pd.read_excel(path, sheet_name="Ménisque", header=1)
    df = _normalise_cols(df)
    df["group"] = "meniscus"
    return df


def load_cyclops(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the Cyclop (cyclops group) sheet and append ``group='cyclops'``.

    Parameters
    ----------
    path : Path, default DATA_PATH
        Excel workbook path.

    Returns
    -------
    pd.DataFrame
        Expected shape ``(100, 20)`` (50 patients × 2 surgeries).
    """
    df = pd.read_excel(path, sheet_name="Cyclop", header=1)
    df = _normalise_cols(df)
    df["group"] = "cyclops"
    return df


def load_both(path: Path = DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience tuple ``(meniscus_df, cyclops_df)``."""
    return load_meniscus(path), load_cyclops(path)


def load_combined(path: Path = DATA_PATH) -> pd.DataFrame:
    """Vertically concatenate both sheets into a single long frame.

    The lesion columns are coerced to ``Int64`` (pandas nullable integer) so
    that the Cyclops NaNs are preserved without promoting Méniscus to float.

    Parameters
    ----------
    path : Path, default DATA_PATH

    Returns
    -------
    pd.DataFrame
        Expected shape ``(138, 20)``.
    """
    m, c = load_both(path)
    combined = pd.concat([m, c], ignore_index=True)
    # Ensure lesion columns are nullable Int64 (Cyclops may carry NaNs)
    from constants import SITES  # local import to avoid cycle at module load

    for col in SITES:
        if col in combined.columns:
            combined[col] = combined[col].astype("Int64")
    # Date columns arrive as DD/MM/YYYY text in the corrected workbook (earlier
    # versions stored them as Excel date cells, which pandas read as Timestamps).
    # Coerce to datetime — dayfirst for the French DD/MM/YYYY notation — so that
    # age/delay derivation and the date-anomaly audit operate on real dates.
    for col in ("date_de_naissance", "date_du_trauma", "date_chir"):
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col], dayfirst=True, errors="coerce")
    return combined


def load_flexum(path: Path = FLEXUM_PATH) -> pd.DataFrame:
    """Load the pre-S2 flexum (extension-deficit) workbook into a long frame.

    The workbook has the same two-sheet layout as the main dataset
    (``Ménisque`` / ``Cyclop``) but a single measurement column
    ``Flexum avant S2`` (header on row 0, unlike the lesion sheets whose header
    is on row 1). Values are signed degrees: ``0`` = full extension,
    ``-5`` = 5° of extension lost (a flexum). Meniscus patients have no cyclops
    nodule, so their flexum is ``0`` throughout.

    Parameters
    ----------
    path : Path, default FLEXUM_PATH
        Excel workbook path (``data/flexum.xlsx``).

    Returns
    -------
    pd.DataFrame
        Columns ``anonyme`` (int), ``group`` (``meniscus`` / ``cyclops``),
        ``flexum_pre_s2`` (float, ≤ 0). One row per patient.
    """
    frames = []
    for sheet, grp in (("Ménisque", "meniscus"), ("Cyclop", "cyclops")):
        sub = pd.read_excel(path, sheet_name=sheet)
        sub = _normalise_cols(sub)
        sub = sub.rename(columns={"flexum_avant_s2": "flexum_pre_s2"})
        sub["anonyme"] = pd.to_numeric(sub["anonyme"], errors="coerce").astype("Int64")
        sub["flexum_pre_s2"] = pd.to_numeric(sub["flexum_pre_s2"], errors="coerce")
        sub["group"] = grp
        frames.append(sub[["anonyme", "group", "flexum_pre_s2"]])
    return pd.concat(frames, ignore_index=True).dropna(subset=["anonyme"])
