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

DATA_PATH: Path = Path(__file__).resolve().parents[2] / "data_paper_cyclops_stats.xlsx"


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
    """Load the Ménisque (control) sheet and append ``group='meniscus'``.

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
    """Load the Cyclop (case) sheet and append ``group='cyclops'``.

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
    from . import SITES  # local import to avoid cycle at module load
    for col in SITES:
        if col in combined.columns:
            combined[col] = combined[col].astype("Int64")
    return combined
