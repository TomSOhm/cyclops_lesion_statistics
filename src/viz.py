"""Visualisation helpers (seaborn / plotly / matplotlib / arviz).

Conventions
-----------
- Group palette: ``Set2`` (Cyclops = orange-ish, Méniscus = green-ish).
- Ordinal lesion scores: ``cividis`` (perceptually uniform).
- Pivot / diverging deltas: ``magma_r``.
- All figures save to ``figures/<notebook_id>_<slug>.png`` (and ``.html``
  for interactive Plotly figures) via :func:`save_fig`.
- Random jitter uses ``RANDOM_SEED`` for reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

# Lazy imports for matplotlib/seaborn so the package remains importable
# even when those backends are missing or broken. Notebooks/tests that need
# plotting will import them locally.
try:
    import matplotlib.pyplot as plt  # noqa: F401
    import seaborn as sns  # noqa: F401

    _HAS_MPL = True
except Exception:  # noqa: BLE001
    plt = None  # type: ignore[assignment]
    sns = None  # type: ignore[assignment]
    _HAS_MPL = False

from constants import HDI_PROB, RANDOM_SEED, SITES

# Default output directory for figures (repo-root/figures/)
FIG_DIR: Path = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Palettes (locked)
GROUP_PALETTE: dict = {"meniscus": "#66c2a5", "cyclops": "#fc8d62"}  # Set2 first two
LESION_CMAP: str = "cividis"
DELTA_CMAP: str = "magma_r"


def set_style() -> None:
    """Apply project-wide matplotlib / seaborn defaults.

    Idempotent  safe to call at the top of every notebook.
    """
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=1.05,
        palette="Set2",
        rc={
            "figure.figsize": (8, 5),
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        },
    )
    np.random.seed(RANDOM_SEED)


def save_fig(fig, nb_id: str, slug: str, html: bool = False) -> Path:
    """Save a figure under ``figures/<nb_id>_<slug>.{png,html}``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
    nb_id : str
        Notebook identifier (e.g. ``"02_progression_total"``).
    slug : str
        Short descriptor (e.g. ``"slopegraph"``).
    html : bool
        If True, also save an HTML version (for Plotly figures).

    Returns
    -------
    Path
        Saved PNG path.
    """
    base = FIG_DIR / f"{nb_id}_{slug}"
    png_path = base.with_suffix(".png")
    # Plotly figures expose write_image / write_html ; matplotlib has savefig.
    if hasattr(fig, "write_image"):
        try:
            fig.write_image(str(png_path), scale=2)
        except Exception:  # noqa: BLE001  kaleido may be missing
            pass
        if html:
            fig.write_html(str(base.with_suffix(".html")))
    else:
        fig.savefig(png_path)
    return png_path


# --- Paired slopegraph (S1 vs S2 lesion_total) ----------------------------


def slopegraph_paired(
    df_wide: pd.DataFrame,
    value_s1: str = "lesion_total_S1",
    value_s2: str = "lesion_total_S2",
    group_col: str = "group",
):
    """Plotly paired slopegraph S1→S2 per patient, coloured by group."""
    import plotly.graph_objects as go

    fig = go.Figure()
    rng = np.random.default_rng(RANDOM_SEED)
    for grp, sub in df_wide.groupby(group_col):
        jitter = rng.normal(0, 0.04, size=len(sub))
        for (_, row), j in zip(sub.iterrows(), jitter):
            fig.add_trace(
                go.Scatter(
                    x=[1 + j, 2 + j],
                    y=[row[value_s1], row[value_s2]],
                    mode="lines+markers",
                    line=dict(color=GROUP_PALETTE.get(grp, "gray"), width=1),
                    marker=dict(size=6, color=GROUP_PALETTE.get(grp, "gray")),
                    name=grp,
                    legendgroup=grp,
                    showlegend=False,
                    opacity=0.55,
                )
            )
    # Dummy traces for legend entries
    for grp in df_wide[group_col].unique():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=GROUP_PALETTE.get(grp, "gray"), width=3),
                name=grp,
                legendgroup=grp,
                showlegend=True,
            )
        )
    fig.update_layout(
        title="Lesion total per patient  S1 → S2",
        xaxis=dict(
            tickmode="array", tickvals=[1, 2], ticktext=["S1", "S2"], title="Surgery"
        ),
        yaxis=dict(title="lesion_total (0–12)"),
        template="plotly_white",
        height=520,
        width=700,
    )
    return fig


# --- Sankey of S1 → S2 transitions on lesion_total -----------------------


def sankey_transitions(
    df_wide: pd.DataFrame,
    value_s1: str = "lesion_total_S1",
    value_s2: str = "lesion_total_S2",
    group: Optional[str] = None,
    group_col: str = "group",
):
    """Plotly Sankey of S1→S2 transitions on a numeric/ordinal column."""
    import plotly.graph_objects as go

    sub = df_wide if group is None else df_wide[df_wide[group_col] == group]
    s1_vals = sub[value_s1].dropna().astype(int)
    s2_vals = sub[value_s2].dropna().astype(int)
    pairs = pd.DataFrame({"s1": s1_vals, "s2": s2_vals}).dropna()
    counts = pairs.groupby(["s1", "s2"]).size().reset_index(name="n")

    s1_labels = sorted(pairs["s1"].unique())
    s2_labels = sorted(pairs["s2"].unique())
    all_labels = [f"S1={v}" for v in s1_labels] + [f"S2={v}" for v in s2_labels]
    label_idx = {l: i for i, l in enumerate(all_labels)}

    source = [label_idx[f"S1={s}"] for s in counts["s1"]]
    target = [label_idx[f"S2={s}"] for s in counts["s2"]]
    value = counts["n"].tolist()

    fig = go.Figure(
        go.Sankey(
            node=dict(label=all_labels, pad=15, thickness=18),
            link=dict(source=source, target=target, value=value),
        )
    )
    title = "S1 → S2 transitions" + (f"  {group}" if group else "")
    fig.update_layout(title=title, height=500, width=700)
    return fig


# --- Heatmap patient × site × time ---------------------------------------


def heatmap_patient_site_time(
    df_long: pd.DataFrame,
    sites: Iterable[str] = tuple(SITES),
    id_col: str = "anonyme",
    group_col: str = "group",
):
    """Two-facet heatmap (S1, S2) of ordinal lesion scores per patient × site.

    Rows are sorted by progression severity (Δlesion_total descending), so
    worst-progressing patients appear at top. A horizontal divider separates
    meniscus from cyclops blocks, and the colorbar carries semantic labels
    (0 = aucune, 1 = partielle, 2 = complète).
    """
    sites = list(sites)
    times = ["S1", "S2"]

    # Compute per-patient Δ for sort order (S2 − S1 sum across sites)
    s_long = df_long.copy()
    s_long["__row_score__"] = s_long[sites].sum(axis=1, min_count=1)
    pivot_scores = s_long.pivot_table(
        index=[group_col, id_col],
        columns="time",
        values="__row_score__",
        aggfunc="first",
    )
    if "S1" in pivot_scores.columns and "S2" in pivot_scores.columns:
        pivot_scores["delta"] = pivot_scores["S2"] - pivot_scores["S1"]
    else:
        pivot_scores["delta"] = 0
    # Sort within each group by delta desc, group order = meniscus first
    order = pivot_scores.reset_index().sort_values(
        [group_col, "delta"], ascending=[True, False]
    )
    order["label"] = order[group_col].str[0].str.upper() + order[id_col].astype(str)
    label_order = order["label"].tolist()
    n_meniscus = int((order[group_col] == "meniscus").sum())

    fig, axes = plt.subplots(
        1, 2, figsize=(8, max(6, 0.18 * len(label_order))), sharey=True
    )
    for ax, t in zip(axes, times):
        sub = df_long[df_long["time"] == t]
        long_sub = sub.melt(
            id_vars=[id_col, group_col],
            value_vars=sites,
            var_name="site",
            value_name="score",
        )
        long_sub["label"] = long_sub[group_col].str[0].str.upper() + long_sub[
            id_col
        ].astype(str)
        mat = long_sub.pivot_table(
            index="label",
            columns="site",
            values="score",
            aggfunc="first",
        )
        mat = mat.reindex(index=label_order, columns=sites)

        is_s2 = t == "S2"
        cbar_kws = dict(label="score", ticks=[0, 1, 2]) if is_s2 else None
        hm = sns.heatmap(
            mat.astype(float),
            ax=ax,
            cmap=LESION_CMAP,
            vmin=0,
            vmax=2,
            cbar=is_s2,
            cbar_kws=cbar_kws,
            linewidths=0.3,
            linecolor="white",
        )
        if is_s2:
            cbar = hm.collections[0].colorbar
            cbar.set_ticks([0, 1, 2])
            cbar.set_ticklabels(["aucune", "partielle", "complète"])
        # Group separator line between meniscus and cyclops blocks
        if 0 < n_meniscus < len(label_order):
            ax.axhline(n_meniscus, color="#555", linestyle="--", linewidth=1.2)
        ax.set_title(t)
        ax.set_xlabel("Compartiment")
    axes[0].set_ylabel("Patient (G=meniscus, C=cyclops, trié par Δ)")
    fig.suptitle(
        "Lésions par patient × compartiment × temps (trié par progression)",
        y=1.02,
    )
    fig.tight_layout()
    return fig


# --- Dumbbell per site (Δ distribution) -----------------------------------


def dumbbell_per_site(
    df_wide: pd.DataFrame,
    sites: Iterable[str] = tuple(SITES),
    group_col: str = "group",
):
    """Plotly small-multiples dumbbell: S1→S2 per patient, faceted by site."""
    import plotly.express as px

    sites = list(sites)
    records = []
    for site in sites:
        s1, s2 = f"{site}_S1", f"{site}_S2"
        if not {s1, s2}.issubset(df_wide.columns):
            continue
        for _, row in df_wide.iterrows():
            records.append({
                "site": site,
                "group": row[group_col],
                "time": "S1",
                "score": row[s1],
                "pid": row["anonyme"],
            })
            records.append({
                "site": site,
                "group": row[group_col],
                "time": "S2",
                "score": row[s2],
                "pid": row["anonyme"],
            })
    long_df = pd.DataFrame(records).dropna(subset=["score"])
    fig = px.strip(
        long_df,
        x="time",
        y="score",
        color="group",
        facet_col="site",
        facet_col_wrap=3,
        color_discrete_map=GROUP_PALETTE,
        category_orders={"time": ["S1", "S2"], "site": sites},
        stripmode="overlay",
    )
    fig.update_traces(jitter=0.3, marker=dict(size=5, opacity=0.6))
    fig.update_layout(
        title="S1 → S2 scores per compartment",
        height=520,
        width=900,
        template="plotly_white",
    )
    return fig


# --- Correlogram (Spearman) ----------------------------------------------


def correlogram(df_patient: pd.DataFrame, cols: Sequence[str]):
    """Seaborn Spearman correlation heatmap on selected patient columns."""
    sub = df_patient[list(cols)].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 2, 0.7 * len(cols) + 1))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman ρ"},
    )
    ax.set_title("Spearman correlogram")
    fig.tight_layout()
    return fig


# --- Ridgeline age by group ----------------------------------------------


def ridgeline_age(
    df_patient: pd.DataFrame, age_col: str = "age_at_trauma", group_col: str = "group"
):
    """Seaborn FacetGrid ridgeline (KDE) of age per group."""
    g = sns.FacetGrid(
        df_patient,
        row=group_col,
        hue=group_col,
        aspect=4,
        height=1.4,
        palette=GROUP_PALETTE,
    )
    g.map(sns.kdeplot, age_col, clip_on=False, fill=True, alpha=0.5, linewidth=1.5)
    g.map(sns.kdeplot, age_col, clip_on=False, color="black", linewidth=1)
    g.set_titles("{row_name}")
    g.set(yticks=[], ylabel="")
    g.despine(left=True)
    g.fig.suptitle(f"{age_col} distribution by group", y=1.02)
    return g.fig


# --- Gantt timeline ------------------------------------------------------


def gantt_timeline(
    df_long: pd.DataFrame,
    max_patients: int = 30,
    id_col: str = "anonyme",
    group_col: str = "group",
):
    """Plotly horizontal bars: trauma → S1 → S2 for first ``max_patients``."""
    import plotly.express as px

    sub = df_long.copy()
    # Pick first max_patients patients per group for readability
    keep = (
        sub
        .groupby(group_col)[id_col]
        .unique()
        .apply(lambda a: list(a)[: max_patients // 2])
        .explode()
        .tolist()
    )
    sub = sub[sub[id_col].isin(keep)]
    sub["patient"] = sub[group_col].str[0].str.upper() + sub[id_col].astype(str)
    long_evt = pd.melt(
        sub,
        id_vars=["patient", group_col],
        value_vars=["date_du_trauma", "date_chir"],
        var_name="event",
        value_name="date",
    ).dropna(subset=["date"])
    fig = px.scatter(
        long_evt,
        x="date",
        y="patient",
        color=group_col,
        symbol="event",
        color_discrete_map=GROUP_PALETTE,
    )
    fig.update_traces(marker=dict(size=9))
    fig.update_layout(
        title="Patient timelines", height=18 * len(keep) + 200, template="plotly_white"
    )
    return fig


# --- Raincloud delays ----------------------------------------------------


def raincloud_delays(
    df_wide: pd.DataFrame, value: str = "inter_surgery_d", group_col: str = "group"
):
    """Raincloud (boxplot + strip + KDE) of a delay column by group.

    Falls back to a violin + strip overlay if ``ptitprince`` is unavailable.
    """
    sub = df_wide[[group_col, value]].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        import ptitprince as pt

        pt.RainCloud(
            x=group_col,
            y=value,
            data=sub,
            palette=GROUP_PALETTE,
            bw=0.2,
            width_viol=0.7,
            ax=ax,
            orient="v",
        )
    except Exception:  # noqa: BLE001  graceful fallback
        sns.violinplot(
            x=group_col,
            y=value,
            data=sub,
            palette=GROUP_PALETTE,
            inner="quartile",
            ax=ax,
            cut=0,
        )
        sns.stripplot(
            x=group_col,
            y=value,
            data=sub,
            color="black",
            alpha=0.5,
            jitter=0.15,
            ax=ax,
            size=4,
        )
    ax.set_title(f"{value} by group")
    fig.tight_layout()
    return fig


# --- Posterior forest plot (ArviZ) ---------------------------------------


def posterior_forest(idata, var_name: str, comp_names: Optional[Sequence[str]] = None):
    """ArviZ forest plot wrapper for posterior coefficients."""
    import arviz as az

    axes = az.plot_forest(
        idata,
        var_names=[var_name],
        combined=True,
        hdi_prob=0.94,
        figsize=(7, max(3, 0.4 * (len(comp_names or [1]) + 2))),
    )
    if hasattr(axes, "__iter__"):
        ax0 = axes[0]
        fig = ax0.figure
    else:
        ax0 = axes
        fig = axes.figure
    try:
        annotate_forest_clinical_bounds(ax0)
    except Exception:  # noqa: BLE001  annotation must never break the figure
        pass
    fig.suptitle(f"Posterior 94% HDI  {var_name}", y=1.02)
    return fig


# --- ArviZ diagnostic wrappers (pass 2) ----------------------------------


def _save_or_return(fig, savefig: Optional[Path | str]) -> "object":
    """Internal: write to disk via :func:`save_fig` if ``savefig`` provided."""
    if savefig is None:
        return fig
    sp = Path(savefig)
    sp.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(fig, "savefig"):
        fig.savefig(sp, dpi=150, bbox_inches="tight")
    return fig


def plot_trace_az(
    idata, var_names: Sequence[str], savefig: Optional[Path | str] = None
):
    """Wrap ``az.plot_trace`` with project-consistent style.

    Parameters
    ----------
    idata : az.InferenceData
    var_names : sequence of str
        Variable names to trace (e.g. ``["delta", "gamma", "beta_c"]``).
    savefig : Path or str, optional
        If provided, saves PNG to this path.
    """
    import arviz as az

    axes = az.plot_trace(
        idata,
        var_names=list(var_names),
        compact=True,
        figsize=(11, 1.6 * len(var_names) + 1),
    )
    fig = axes.ravel()[0].figure if hasattr(axes, "ravel") else axes.figure
    fig.suptitle(f"Trace  {', '.join(var_names)}", y=1.02)
    fig.tight_layout()
    return _save_or_return(fig, savefig)


def plot_pair_az(idata, var_names: Sequence[str], savefig: Optional[Path | str] = None):
    """Wrap ``az.plot_pair`` for top-level scalar parameters (kde + scatter)."""
    import arviz as az

    axes = az.plot_pair(
        idata,
        var_names=list(var_names),
        kind=["scatter", "kde"],
        marginals=True,
        divergences=True,
    )
    fig = axes[0, 0].figure if hasattr(axes, "shape") else axes.figure
    fig.suptitle(f"Pair plot  {', '.join(var_names)}", y=1.02)
    return _save_or_return(fig, savefig)


def plot_ppc_az(
    idata,
    observed: Optional[str] = None,
    savefig: Optional[Path | str] = None,
    **kwargs,
):
    """Wrap ``az.plot_ppc``.

    Requires ``idata.posterior_predictive`` (use
    :func:`bayes_models.ppc_m3` or ``pm.sample_posterior_predictive``).
    """
    import arviz as az

    # ``data_pairs`` was removed in ArviZ ≥ 1.0 in favour of automatic
    # group/var matching. Try the modern call first; fall back to legacy.
    try:
        ax = az.plot_ppc(idata, **kwargs)
    except TypeError:
        ax = az.plot_ppc(
            idata,
            data_pairs=({observed: observed} if observed else None),
            **kwargs,
        )
    fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
    fig.suptitle("Posterior predictive check", y=1.02)
    return _save_or_return(fig, savefig)


def plot_energy_az(idata, savefig: Optional[Path | str] = None):
    """Wrap ``az.plot_energy`` (NUTS energy / BFMI diagnostic).

    Bimodality or marginal/E_T separation flags potential funnel / divergence
    issues  see escalation table in [[02.5-modeles-bayesiens]] §7.
    """
    import arviz as az

    ax = az.plot_energy(idata)
    fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
    fig.suptitle("NUTS energy diagnostic", y=1.02)
    return _save_or_return(fig, savefig)


def plot_shrinkage(
    no_pooling_est: Sequence[float],
    partial_pooling_est: Sequence[float],
    names: Sequence[str],
    savefig: Optional[Path | str] = None,
):
    """Side-by-side scatter showing shrinkage from no-pooling to partial-pooling.

    Plots two columns of points (no-pool on x=0, partial-pool on x=1) with
    connecting segments labelled by ``names``. The shorter the segment, the
    weaker the shrinkage; segments collapsing toward the partial-pool grand
    mean illustrate the hierarchical pull.

    Parameters
    ----------
    no_pooling_est : sequence of float
        Per-compartment estimate without pooling (e.g. independent MLEs).
    partial_pooling_est : sequence of float
        Per-compartment posterior mean from the hierarchical fit.
    names : sequence of str
        Labels (e.g. ``SITES``).
    savefig : Path or str, optional
    """
    no_pool = np.asarray(no_pooling_est, dtype=float)
    partial = np.asarray(partial_pooling_est, dtype=float)
    if len(no_pool) != len(partial) or len(no_pool) != len(names):
        raise ValueError("no_pool, partial, and names must have identical length.")

    fig, ax = plt.subplots(figsize=(6, 4 + 0.15 * len(names)))
    for nm, np_v, pp_v in zip(names, no_pool, partial):
        ax.plot([0, 1], [np_v, pp_v], "-", color="#888", alpha=0.6)
        ax.text(-0.05, np_v, nm, ha="right", va="center", fontsize=9)
        ax.text(1.05, pp_v, nm, ha="left", va="center", fontsize=9)
    ax.scatter(
        [0] * len(no_pool), no_pool, color="#fc8d62", s=60, label="no pooling", zorder=3
    )
    ax.scatter(
        [1] * len(partial),
        partial,
        color="#66c2a5",
        s=60,
        label="partial pooling",
        zorder=3,
    )
    ax.axhline(
        partial.mean(),
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"partial-pool grand mean = {partial.mean():.2f}",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no pooling", "partial pooling"])
    ax.set_ylabel("Estimate")
    ax.set_title("Shrinkage: no-pooling → partial pooling")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(-0.3, 1.3)
    fig.tight_layout()
    return _save_or_return(fig, savefig)


def plot_cutpoints(idata, var_name: str = "cut", savefig: Optional[Path | str] = None):
    """Ordered cutpoints with HDI bands (M3 proportional-odds visual).

    Plots each ``cut[k]`` posterior mean with a horizontal HDI 94 % band.
    A stable gap between consecutive cutpoints supports the proportional-odds
    assumption  see :func:`bayes_models.proportional_odds_check`.
    """
    import arviz as az

    post = idata.posterior[var_name]
    means = post.mean(dim=("chain", "draw")).values
    hdi = az.hdi(post, hdi_prob=0.94)[var_name].values  # (cut, [lo, hi])

    fig, ax = plt.subplots(figsize=(6, 3))
    for k, (m, (lo, hi)) in enumerate(zip(means, hdi)):
        ax.errorbar(
            m, k, xerr=[[m - lo], [hi - m]], fmt="o", color="#3b8bba", capsize=4
        )
        ax.text(m, k + 0.12, f"cut[{k}] = {m:+.2f}", ha="center", fontsize=9)
    ax.axvline(0, color="grey", linestyle=":", alpha=0.6)
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels([f"cut[{k}]" for k in range(len(means))])
    ax.set_xlabel("Latent logit scale")
    ax.set_title(f"Ordered cutpoints (HDI 94%)  {var_name}")
    fig.tight_layout()
    return _save_or_return(fig, savefig)


# --- Interpretability helpers (I4–I8) ------------------------------------


def slopegraph_paired_annotated(
    df_wide: pd.DataFrame,
    value_s1: str = "lesion_total_S1",
    value_s2: str = "lesion_total_S2",
    group_col: str = "group",
):
    """Plotly faceted slopegraph S1→S2 with median trace + worsening %.

    Two-column subplot, one per group. Each patient drawn as a thin
    semi-transparent line; the group median trajectory is overlaid as a
    bold line. The proportion of patients worsening (Δ > 0) is annotated
    in the top-right corner of each facet.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    groups = list(df_wide[group_col].dropna().unique())
    fig = make_subplots(
        rows=1,
        cols=len(groups),
        subplot_titles=[g.capitalize() for g in groups],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    rng = np.random.default_rng(RANDOM_SEED)
    for col_i, grp in enumerate(groups, start=1):
        sub = df_wide[df_wide[group_col] == grp].dropna(subset=[value_s1, value_s2])
        jitter = rng.normal(0, 0.04, size=len(sub))
        for (_, row), j in zip(sub.iterrows(), jitter):
            fig.add_trace(
                go.Scatter(
                    x=[1 + j, 2 + j],
                    y=[row[value_s1], row[value_s2]],
                    mode="lines+markers",
                    line=dict(color=GROUP_PALETTE.get(grp, "gray"), width=1),
                    marker=dict(size=5, color=GROUP_PALETTE.get(grp, "gray")),
                    opacity=0.45,
                    showlegend=False,
                ),
                row=1,
                col=col_i,
            )
        # Median trajectory (bold)
        med_s1 = float(sub[value_s1].median())
        med_s2 = float(sub[value_s2].median())
        fig.add_trace(
            go.Scatter(
                x=[1, 2],
                y=[med_s1, med_s2],
                mode="lines+markers",
                line=dict(color="black", width=4),
                marker=dict(size=10, color="black", symbol="diamond"),
                name=f"{grp} median",
                showlegend=(col_i == 1),
            ),
            row=1,
            col=col_i,
        )
        # Worsening %
        deltas = sub[value_s2] - sub[value_s1]
        pct_worse = float((deltas > 0).mean() * 100.0)
        pct_stable = float((deltas == 0).mean() * 100.0)
        pct_better = float((deltas < 0).mean() * 100.0)
        fig.add_annotation(
            x=1.5,
            y=sub[[value_s1, value_s2]].max().max() * 1.02,
            text=(
                f"↑ {pct_worse:.0f}%  = {pct_stable:.0f}%  "
                f"↓ {pct_better:.0f}% (n = {len(sub)})"
            ),
            showarrow=False,
            font=dict(size=11),
            row=1,
            col=col_i,
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=[1, 2],
            ticktext=["S1", "S2"],
            row=1,
            col=col_i,
        )
    fig.update_layout(
        title=f"Trajectoires S1 → S2 par groupe ({value_s1.replace('_S1', '')})",
        yaxis_title="lesion_total (0–12)",
        template="plotly_white",
        height=520,
        width=900,
    )
    return fig


def sankey_transitions_normalized(
    df_wide: pd.DataFrame,
    value_s1: str = "lesion_total_S1",
    value_s2: str = "lesion_total_S2",
    group_col: str = "group",
):
    """Side-by-side Sankeys per group, weighted by within-group %.

    Solves the visual imbalance of absolute counts (cyclops dominates
    meniscus). Each Sankey's link widths are scaled to 100% within group
    so behavioural patterns (worsening fractions) compare directly.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    groups = list(df_wide[group_col].dropna().unique())
    fig = make_subplots(
        rows=1,
        cols=len(groups),
        specs=[[{"type": "sankey"} for _ in groups]],
        subplot_titles=[
            f"{g.capitalize()} (n={int((df_wide[group_col] == g).sum())})"
            for g in groups
        ],
        horizontal_spacing=0.05,
    )
    for col_i, grp in enumerate(groups, start=1):
        sub = df_wide[df_wide[group_col] == grp]
        pairs = pd.DataFrame({
            "s1": sub[value_s1].dropna().astype(int),
            "s2": sub[value_s2].dropna().astype(int),
        }).dropna()
        if pairs.empty:
            continue
        counts = pairs.groupby(["s1", "s2"]).size().reset_index(name="n")
        total = counts["n"].sum()
        counts["pct"] = 100.0 * counts["n"] / total
        s1_labels = sorted(pairs["s1"].unique())
        s2_labels = sorted(pairs["s2"].unique())
        all_labels = [f"S1={v}" for v in s1_labels] + [f"S2={v}" for v in s2_labels]
        label_idx = {l: i for i, l in enumerate(all_labels)}
        source = [label_idx[f"S1={s}"] for s in counts["s1"]]
        target = [label_idx[f"S2={s}"] for s in counts["s2"]]
        value = counts["pct"].tolist()
        fig.add_trace(
            go.Sankey(
                node=dict(
                    label=all_labels,
                    pad=12,
                    thickness=14,
                    color=GROUP_PALETTE.get(grp, "gray"),
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    customdata=counts["n"].tolist(),
                    hovertemplate="%{value:.1f}% (n = %{customdata})<extra></extra>",
                ),
            ),
            row=1,
            col=col_i,
        )
    fig.update_layout(
        title="Transitions S1 → S2 normalisées intra-groupe (%)",
        height=500,
        width=1000,
    )
    return fig


def annotate_forest_clinical_bounds(
    ax,
    rope: tuple[float, float] = (-0.1, 0.1),
    cohen_small: float = 0.2,
    cohen_medium: float = 0.5,
):
    """Shade ROPE band + small / medium effect bands on a forest axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis with x = effect size.
    rope : (float, float)
        Practical equivalence band.
    cohen_small, cohen_medium : float
        Cohen effect-size band boundaries (symmetric).
    """
    ax.axvspan(
        rope[0],
        rope[1],
        color="#fdd",
        alpha=0.4,
        label=f"ROPE [{rope[0]:+.2f}, {rope[1]:+.2f}]",
    )
    for sign in (-1, +1):
        ax.axvspan(sign * cohen_small, sign * cohen_medium, color="#cfe8c2", alpha=0.25)
        ax.axvspan(
            sign * cohen_medium,
            sign * (cohen_medium + 0.3),
            color="#9fcb86",
            alpha=0.20,
        )
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)


def make_clinical_summary_card(
    wide: pd.DataFrame,
    delta_col: str = "delta_lesion_total",
    group_col: str = "group",
    idata=None,
    bayes_var: Optional[str] = None,
    rope: tuple[float, float] = (-0.1, 0.1),
    savefig: Optional[Path | str] = None,
):
    """One 2×2 figure summarising the group comparison for a manuscript.

    Panels:
      (0,0) violin + box of Δ per group (+ sample size labels)
      (0,1) histogram + KDE of Δ per group (semi-transparent overlay)
      (1,0) Cliff's δ point estimate + BCa CI with ROPE / Cohen bands
      (1,1) posterior of ``bayes_var`` from ``idata`` (if provided), with
            ROPE shaded and p_direction annotated; else a textual placeholder.

    Parameters
    ----------
    wide : pd.DataFrame
        Output of ``preprocessing.to_wide`` (one row per patient).
    delta_col : str
        Δ column to summarise (default: ``delta_lesion_total``).
    group_col : str
    idata : az.InferenceData, optional
    bayes_var : str, optional
        Variable to plot in panel (1,1); required if ``idata`` is given.
    rope : (float, float)
    savefig : Path or str, optional
    """
    import tests_freq as tf

    sub = wide[[group_col, delta_col]].dropna()
    groups = sorted(sub[group_col].unique())
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups for clinical card, got {groups!r}.")
    g_a, g_b = groups
    a = sub.loc[sub[group_col] == g_a, delta_col].astype(float).values
    b = sub.loc[sub[group_col] == g_b, delta_col].astype(float).values

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (0,0) violin + box
    sns.violinplot(
        x=group_col,
        y=delta_col,
        data=sub,
        ax=axes[0, 0],
        palette=GROUP_PALETTE,
        inner="box",
        cut=0,
    )
    sns.stripplot(
        x=group_col,
        y=delta_col,
        data=sub,
        ax=axes[0, 0],
        color="black",
        alpha=0.4,
        jitter=0.15,
        size=4,
    )
    axes[0, 0].axhline(0, color="grey", linestyle=":")
    axes[0, 0].set_title(f"Δ {delta_col} par groupe (n={len(a)} vs {len(b)})")

    # (0,1) histogram overlay
    for grp, arr in [(g_a, a), (g_b, b)]:
        axes[0, 1].hist(
            arr,
            bins=15,
            alpha=0.55,
            density=True,
            color=GROUP_PALETTE.get(grp, "gray"),
            label=grp,
        )
    axes[0, 1].axvline(0, color="grey", linestyle=":")
    axes[0, 1].set_title(f"Distribution Δ {delta_col}")
    axes[0, 1].set_xlabel("Δ")
    axes[0, 1].set_ylabel("Densité")
    axes[0, 1].legend()

    # (1,0) Cliff's δ + CI
    res = tf.mwu_with_effects(a, b, n_boot=2000)
    d = res["cliffs_delta"]
    lo, hi = res["delta_ci_lo"], res["delta_ci_hi"]
    axes[1, 0].errorbar(
        d,
        0,
        xerr=[[d - lo], [hi - d]],
        fmt="o",
        color="#3b8bba",
        capsize=4,
        markersize=10,
        label=f"Cliff's δ = {d:+.2f}",
    )
    annotate_forest_clinical_bounds(axes[1, 0], rope=rope)
    axes[1, 0].set_xlim(-1, 1)
    axes[1, 0].set_yticks([])
    axes[1, 0].set_title(
        f"Cliff's δ ({res.get('cliffs_delta_magnitude', '?')})  p = {res['pvalue']:.3f}"
    )
    axes[1, 0].set_xlabel("Cliff's δ")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    # (1,1) posterior or placeholder
    if idata is not None and bayes_var is not None:
        vals = idata.posterior[bayes_var].values.ravel()
        axes[1, 1].hist(vals, bins=40, density=True, color="#7aa6c2", alpha=0.85)
        axes[1, 1].axvspan(rope[0], rope[1], color="#fdd", alpha=0.5)
        axes[1, 1].axvline(0, color="black", linestyle=":")
        p_above = float((vals > 0).mean())
        pd_val = max(p_above, 1 - p_above)
        axes[1, 1].set_title(f"Posterior {bayes_var}  pd = {pd_val:.2%}")
        axes[1, 1].set_xlabel(bayes_var)
        axes[1, 1].set_ylabel("Densité")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Posterior bayésien\n(idata non fourni)",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=11,
            color="#888",
        )
        axes[1, 1].set_axis_off()

    fig.suptitle(
        f"Synthèse clinique : {g_a} vs {g_b} ({delta_col})",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    return _save_or_return(fig, savefig)


# ===========================================================================
# PUBLICATION FIGURES (English, 300 dpi, colorblind-safe Set2)
# ---------------------------------------------------------------------------
# Functions below back the manuscript figure manifest (make_figures.py). They
# are self-contained matplotlib figures with English titles/axes/legends, a
# locked Cyclops-vs-Meniscus Set2 palette, and a fixed 300 dpi raster export.
# They never mutate input frames and read every number from the analysis
# artefacts (results.json / idata_*.nc / the preprocessing pipeline)  no
# value is hard-coded.
# ===========================================================================

# Display labels (capitalised, English) for the two cohorts.
GROUP_LABELS: dict = {"cyclops": "Cyclops", "meniscus": "Meniscus"}
# Canonical group order for plotting (cyclops first, then meniscus).
GROUP_ORDER: list[str] = ["cyclops", "meniscus"]

# Pretty compartment labels for axes (native French anatomical names → EN).
# French laterality: E = externe (lateral), I = interne (medial). The codes are
# kept in parentheses so a reviewer can cross-check against the data dictionary.
#   trochlée → Trochlea,  rotule → Patella           (PF block)
#   PTE/PTI  → lateral/medial tibial plateau          (FT block)
#   CFE/CFI  → lateral/medial femoral condyle         (FT block)
SITE_LABELS: dict = {
    "trochlée": "Trochlea",
    "rotule": "Patella",
    "pte": "Lateral\ntibial plateau",
    "pti": "Medial\ntibial plateau",
    "cfe": "Lateral\nfemoral condyle",
    "cfi": "Medial\nfemoral condyle",
}

PUB_DPI: int = 300


def set_pub_style() -> None:
    """Apply the publication matplotlib/seaborn theme (300 dpi, clean serif-grid).

    Idempotent. Distinct from :func:`set_style` (notebook context, 150 dpi):
    this sets the manuscript defaults used by ``make_figures.py``  larger
    fonts, 300 dpi raster export, tight bounding box, and the locked Set2
    palette. ``DejaVu Sans`` carries the Greek/maths glyphs (δ, μ, ±, ≥) used
    in captions, so no LaTeX is required.
    """
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.25,
        palette="Set2",
        rc={
            "figure.dpi": 110,
            "savefig.dpi": PUB_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#444",
            "grid.color": "#dddddd",
            "figure.autolayout": False,
        },
    )
    np.random.seed(RANDOM_SEED)


def save_pub_fig(fig, name: str, fig_dir: Optional[Path | str] = None) -> Path:
    """Save a matplotlib figure as ``figures/<name>.png`` at 300 dpi.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        File stem (the manifest name, e.g. ``"fig2_pf_progression"``).
    fig_dir : Path or str, optional
        Output directory (defaults to the package ``FIG_DIR``).

    Returns
    -------
    Path
        The written PNG path.
    """
    out_dir = Path(fig_dir) if fig_dir is not None else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=PUB_DPI, bbox_inches="tight")
    return png


def _group_color(grp: str) -> str:
    return GROUP_PALETTE.get(grp, "#999999")


# --- fig1 : baseline balance Love plot ------------------------------------


def love_plot_smd(
    smd_rows: Sequence[tuple[str, float]],
    baseline_score_p: Optional[float] = None,
    thresholds: tuple[float, float] = (0.1, 0.25),
    title: str = "Baseline covariate balance (standardised mean differences)",
):
    """Love plot of baseline SMDs (Cyclops − Meniscus) with balance bands.

    Parameters
    ----------
    smd_rows : sequence of (label, smd)
        Covariate label and signed SMD (Cyclops minus Meniscus). Drawn sorted
        by absolute magnitude (largest imbalance at top).
    baseline_score_p : float, optional
        If given, annotates the equivalence of the S1 lesion score (e.g.
        ``p = 0.78``) as a caption  the outcome itself is balanced even
        though some covariates are not.
    thresholds : (float, float)
        Vertical guide lines at |SMD| = 0.10 (negligible) and 0.25
        (Austin's "meaningful imbalance" cut).
    title : str
    """
    rows = sorted(
        smd_rows, key=lambda kv: abs(kv[1])
    )  # smallest first → top is largest
    labels = [r[0] for r in rows]
    vals = [float(r[1]) for r in rows]
    y = np.arange(len(rows))

    # Taller than before + a reserved empty band at the bottom (see set_ylim)
    # so the lower-right legend never sits on top of the bottom data point.
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(rows) + 2.6))
    # Balance bands.
    t1, t2 = thresholds
    ax.axvspan(-t1, t1, color="#cfe8c2", alpha=0.45, zorder=0)
    for thr, ls in ((t1, ":"), (t2, "--")):
        for sign in (-1, 1):
            ax.axvline(sign * thr, color="#888", linestyle=ls, linewidth=1.0, zorder=1)
    ax.axvline(0, color="#222", linewidth=1.1, zorder=2)

    # Colour points red when |SMD| ≥ 0.25 (meaningful imbalance), else neutral.
    colors = ["#d1495b" if abs(v) >= t2 else "#3b6ea5" for v in vals]
    ax.scatter(vals, y, s=110, color=colors, zorder=3, edgecolor="white", linewidth=0.8)
    for yi, v in zip(y, vals):
        ax.annotate(
            f"{v:+.2f}",
            (v, yi),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Standardised mean difference (Cyclops − Meniscus)")
    ax.set_title(title)
    xmax = max(0.35, max(abs(v) for v in vals) * 1.25)
    ax.set_xlim(-xmax, xmax)
    # Extra headroom below row 0 → the legend gets its own band, clear of points.
    ax.set_ylim(-1.25, len(rows) - 0.4)

    # Threshold legend.
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0], [0], color="#888", linestyle=":", label="|SMD| = 0.10 (negligible)"
        ),
        Line2D(
            [0], [0], color="#888", linestyle="--", label="|SMD| = 0.25 (meaningful)"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d1495b",
            markersize=9,
            label="|SMD| ≥ 0.25",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)

    if baseline_score_p is not None:
        ax.text(
            0.02,
            0.02,
            f"Baseline S1 lesion score: equivalent\n(Mann–Whitney p = {baseline_score_p:.2f})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f9f0", ec="#9fcb86"),
        )
    fig.tight_layout()
    return fig


# --- fig2 : primary patellofemoral progression (raincloud) ----------------


def raincloud_progression(
    df_wide: pd.DataFrame,
    value_col: str = "delta_lesion_pf",
    group_col: str = "group",
    worsened_pct: Optional[dict] = None,
    cliff_delta: Optional[float] = None,
    perm_p: Optional[float] = None,
    title: str = "Patellofemoral progression (primary)",
    ylabel: str = "Δ patellofemoral lesion score (S2 − S1)",
):
    """Half-violin + box + jittered points (raincloud) of Δ per group.

    The PRIMARY figure. Each group gets a one-sided KDE ("cloud"), a slim
    boxplot, and rain (jittered raw points). Annotates the proportion
    worsening per group and, if supplied, the Cliff δ + permutation p.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Output of :func:`preprocessing.to_wide`.
    value_col : str
        Δ column to display (default PF block delta).
    worsened_pct : dict, optional
        ``{group: percent_worsening}``; if None, computed from ``value_col``.
    cliff_delta, perm_p : float, optional
        Effect size + permutation p for the stats banner.
    """
    sub = df_wide[[group_col, value_col]].dropna().copy()
    groups = [g for g in GROUP_ORDER if g in sub[group_col].unique()]
    rng = np.random.default_rng(RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for i, grp in enumerate(groups):
        vals = sub.loc[sub[group_col] == grp, value_col].astype(float).values
        color = _group_color(grp)
        # Half-violin (cloud) offset to the left of the category centre.
        try:
            vp = ax.violinplot(
                vals,
                positions=[i],
                widths=0.8,
                showextrema=False,
                showmedians=False,
            )
            for b in vp["bodies"]:
                # Clip to left half.
                verts = b.get_paths()[0].vertices
                verts[:, 0] = np.clip(verts[:, 0], -np.inf, i)
                b.set_facecolor(color)
                b.set_edgecolor("#555")
                b.set_alpha(0.55)
        except Exception:  # noqa: BLE001  degenerate (all-equal) violins
            pass
        # Slim boxplot at the centre.
        bp = ax.boxplot(
            vals,
            positions=[i],
            widths=0.12,
            vert=True,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.6),
            boxprops=dict(facecolor="white", edgecolor="#333"),
            whiskerprops=dict(color="#333"),
            capprops=dict(color="#333"),
            manage_ticks=False,
        )
        # Rain: jittered points to the right.
        jitter = rng.uniform(0.06, 0.30, size=len(vals))
        ax.scatter(
            i + jitter,
            vals + rng.normal(0, 0.04, size=len(vals)),
            s=26,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
        )
        # Worsened % annotation above each group.
        if worsened_pct is not None and grp in worsened_pct:
            pct = worsened_pct[grp]
        else:
            pct = float((vals > 0).mean() * 100.0)
        ax.text(
            i,
            ax.get_ylim()[1],
            f"worsened\n{pct:.0f}%  (n={len(vals)})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    ax.axhline(0, color="#777", linestyle=":", linewidth=1.1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in groups])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Headroom for the worsened-% labels.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))

    # Stats banner.
    bits = []
    if cliff_delta is not None:
        bits.append(f"Cliff δ = {cliff_delta:+.2f}")
    if perm_p is not None:
        p_txt = "< 0.001" if perm_p < 0.001 else f"= {perm_p:.3f}"
        bits.append(f"permutation p {p_txt}")
    if bits:
        ax.text(
            0.5,
            -0.16,
            "  •  ".join(bits),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            color="#222",
        )
    fig.tight_layout()
    return fig


# --- fig3 : per-compartment worsening (PF vs FT blocks) -------------------


def per_compartment_bars(
    per_comp: pd.DataFrame,
    title: str = "Per-compartment worsening: patellofemoral signal vs femorotibial dilution",
):
    """Grouped bars of worsening % per compartment, blocked PF vs FT.

    Parameters
    ----------
    per_comp : pd.DataFrame
        ``per_compartment.csv`` with columns ``compartment, block,
        worsened_pct_cyc, worsened_pct_men`` (and optionally ``cliff_delta``).
    """
    df = per_comp.copy()
    # Order: PF block first (trochlée, rotule), then FT block.
    block_rank = {"PF": 0, "FT": 1}
    df["__brank"] = df["block"].map(block_rank).fillna(9)
    df = df.sort_values(["__brank", "compartment"], kind="stable").reset_index(
        drop=True
    )

    labels = [SITE_LABELS.get(c, c) for c in df["compartment"]]
    x = np.arange(len(df))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(
        x - w / 2,
        df["worsened_pct_cyc"],
        width=w,
        color=_group_color("cyclops"),
        edgecolor="#555",
        label="Cyclops",
    )
    ax.bar(
        x + w / 2,
        df["worsened_pct_men"],
        width=w,
        color=_group_color("meniscus"),
        edgecolor="#555",
        label="Meniscus",
    )

    # Value labels on bars.
    for xi, vc, vm in zip(x, df["worsened_pct_cyc"], df["worsened_pct_men"]):
        ax.text(
            xi - w / 2, vc + 1.0, f"{vc:.0f}", ha="center", va="bottom", fontsize=8.5
        )
        ax.text(
            xi + w / 2, vm + 1.0, f"{vm:.0f}", ha="center", va="bottom", fontsize=8.5
        )

    # Block separator + block labels.
    n_pf = int((df["block"] == "PF").sum())
    if 0 < n_pf < len(df):
        sep = n_pf - 0.5
        ax.axvline(sep, color="#444", linestyle="--", linewidth=1.3)
        ymax = max(df["worsened_pct_cyc"].max(), df["worsened_pct_men"].max())
        ax.text(
            (n_pf - 1) / 2,
            ymax * 1.12,
            "Patellofemoral (PF)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#3b3b6b",
        )
        ax.text(
            (n_pf + len(df) - 1) / 2,
            ymax * 1.12,
            "Femorotibial (FT)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#6b3b3b",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Patients worsening (%)")
    ax.set_xlabel("Compartment")
    ax.set_ylim(
        0, max(df["worsened_pct_cyc"].max(), df["worsened_pct_men"].max()) * 1.28 + 5
    )
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


# --- fig4 : topographic specificity within cyclops (paired PF vs FT) ------


def topographic_specificity(
    df_wide: pd.DataFrame,
    pf_col: str = "delta_lesion_pf",
    ft_col: str = "delta_lesion_ft",
    group_col: str = "group",
    cyclops: str = "cyclops",
    n_pf_worsened: Optional[int] = None,
    n_ft_worsened: Optional[int] = None,
    rank_biserial: Optional[float] = None,
    wilcoxon_p: Optional[float] = None,
    title: str = "Topographic specificity within cyclops (paired ΔPF vs ΔFT)",
):
    """Count-bubble grid of per-patient (ΔPF, ΔFT) in the cyclops group.

    Each cyclops patient contributes one (ΔPF, ΔFT) pair; identical pairs are
    pooled into a **bubble** placed at that integer cell, its **area ∝ the number
    of patients** and the count printed inside. Bubbles are coloured by region:
    green = **PF-specific** (ΔPF > 0, ΔFT ≤ 0), red = **FT involved** (ΔFT > 0),
    grey = neither. The ``ΔPF = ΔFT`` diagonal and the zero axes are drawn for
    reference. The whole story  *the damage lands in PF, FT is spared*  reads
    in one glance: the mass hugs the ΔFT = 0 row at ΔPF > 0.

    Replaces the earlier paired-slope spaghetti (unreadable line crossings).

    Parameters
    ----------
    df_wide : pd.DataFrame
    pf_col, ft_col : str
        Per-patient block-Δ columns.
    cyclops : str
        Group to restrict to (default cyclops).
    n_pf_worsened, n_ft_worsened, rank_biserial, wilcoxon_p : optional
        Stats for the banner (from :func:`tests_freq.paired_pf_vs_ft`).
    """
    from collections import Counter

    sub = df_wide[df_wide[group_col] == cyclops][[pf_col, ft_col]].dropna()
    pf = sub[pf_col].round().astype(int).values
    ft = sub[ft_col].round().astype(int).values
    n = len(sub)
    counts = Counter(zip(pf.tolist(), ft.tolist()))

    fig, ax = plt.subplots(figsize=(7.4, 5.8), layout="constrained")

    lo = min(pf.min(), ft.min(), 0) - 0.7
    hi = max(pf.max(), ft.max(), 1) + 0.9
    # Shade the PF-specific region (ΔPF > 0, ΔFT ≤ 0).
    ax.axhspan(lo, 0, xmin=0, xmax=1, color="#e8f3e6", zorder=0)
    # Identity diagonal + zero axes.
    ax.plot(
        [lo, hi],
        [lo, hi],
        ls="--",
        color="#bbbbbb",
        lw=1.1,
        zorder=1,
        label="ΔPF = ΔFT (equal worsening)",
    )
    ax.axhline(0, color="#777", ls=":", lw=1.0, zorder=1)
    ax.axvline(0, color="#777", ls=":", lw=1.0, zorder=1)

    for (x, yv), k in counts.items():
        if x > 0 and yv <= 0:
            color = "#1b7837"  # PF-specific
        elif yv > 0:
            color = "#d1495b"  # FT involved
        else:
            color = "#9aa7b3"  # neither
        ax.scatter(
            x,
            yv,
            s=90 + 130 * k,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        ax.text(
            x,
            yv,
            str(k),
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    pf_specific = sum(k for (x, yv), k in counts.items() if x > 0 and yv <= 0)
    both = sum(k for (x, yv), k in counts.items() if x > 0 and yv > 0)
    neither = sum(k for (x, yv), k in counts.items() if x <= 0 and yv <= 0)
    if n_pf_worsened is None:
        n_pf_worsened = int((pf > 0).sum())
    if n_ft_worsened is None:
        n_ft_worsened = int((ft > 0).sum())

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ΔPF  patellofemoral worsening (S2 − S1)")
    ax.set_ylabel("ΔFT  femorotibial worsening (S2 − S1)")

    # Region key + counts box (top-left, clear of the data mass at bottom-right).
    txt = (
        f"PF-specific (ΔPF>0, ΔFT≤0): {pf_specific}\n"
        f"FT also worsens (ΔFT>0): {n_ft_worsened}\n"
        f"neither: {neither}   ·   n = {n}"
    )
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec="#bbbbbb"),
        zorder=5,
    )

    banner = [
        f"Within-patient: {n_pf_worsened}/{n} worsen PF, only {n_ft_worsened}/{n} worsen FT"
    ]
    stat_bits = []
    if rank_biserial is not None:
        stat_bits.append(f"matched-pairs r = {rank_biserial:+.2f}")
    if wilcoxon_p is not None:
        p_txt = "< 0.001" if wilcoxon_p < 0.001 else f"= {wilcoxon_p:.3f}"
        stat_bits.append(f"Wilcoxon p {p_txt}")
    if stat_bits:
        banner.append("  •  ".join(stat_bits))
    ax.set_title("\n".join(banner), fontsize=9, color="#555", pad=4)
    fig.suptitle(title, fontsize=12.5, fontweight="bold")
    ax.legend(loc="lower right", framealpha=1.0, fontsize=8.5, edgecolor="#cccccc")
    return fig


# --- fig5 : M3 posterior forest -------------------------------------------


def forest_m3(
    idata,
    var_names: Sequence[str] = ("delta_pf", "delta_ft", "contrast_pf_ft", "gamma"),
    labels: Optional[dict] = None,
    hdi_prob: float = HDI_PROB,
    title: str = "M3 posterior effects (94% HDI)",
):
    """Forest plot of selected M3 scalar posteriors with a reference line at 0.

    Parameters
    ----------
    idata : az.InferenceData
        M3 posterior (``idata_m3.nc``).
    var_names : sequence of str
        Scalar variables to display.
    labels : dict, optional
        ``{var_name: pretty_label}`` for the y axis.
    hdi_prob : float
        HDI mass (default 0.94, contract).
    """
    import arviz as az

    labels = labels or {
        "delta_pf": "δ PF (patellofemoral)",
        "delta_ft": "δ FT (femorotibial)",
        "contrast_pf_ft": "Contrast PF − FT",
        "gamma": "γ (time main effect)",
    }
    present = [v for v in var_names if v in idata.posterior.data_vars]

    def _hdi_interval(samples: np.ndarray) -> tuple[float, float]:
        """HDI of a 1-D sample array, robust across ArviZ API versions."""
        for kw in ("prob", "hdi_prob"):
            try:
                r = np.asarray(az.hdi(samples, **{kw: hdi_prob}))
                return float(r.ravel()[0]), float(r.ravel()[1])
            except TypeError:
                continue
        # Last-resort: equal-tailed interval.
        a = (1 - hdi_prob) / 2
        return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))

    fig, ax = plt.subplots(figsize=(7.2, 0.7 * len(present) + 1.6))
    post = idata.posterior
    y = np.arange(len(present))[::-1]  # first var at top
    for yi, v in zip(y, present):
        vals = np.asarray(post[v].values).ravel()
        mean = float(vals.mean())
        lo, hi = _hdi_interval(vals)
        p_gt0 = float((vals > 0).mean())
        color = "#2a7f62" if (lo > 0 or hi < 0) else "#b08900"
        ax.plot(
            [lo, hi],
            [yi, yi],
            color=color,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            [mean], [yi], s=85, color=color, edgecolor="white", linewidth=0.9, zorder=3
        )
        ax.annotate(
            f"{mean:+.2f}  [{lo:+.2f}, {hi:+.2f}]   P(>0)={p_gt0:.2f}",
            (hi, yi),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color="#333",
        )

    ax.axvline(0, color="#222", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(v, v) for v in present])
    ax.set_xlabel("Posterior effect on the latent logit scale (S2 − S1 contrast)")
    ax.set_title(f"{title}")
    # Headroom on the right for the annotations.
    xlo, xhi = ax.get_xlim()
    ax.set_xlim(xlo - 0.2 * (xhi - xlo), xhi + 0.55 * (xhi - xlo))
    ax.set_ylim(-0.6, len(present) - 0.4)
    fig.tight_layout()
    return fig


# --- fig6 : M3 convergence diagnostics ------------------------------------


def diagnostics_m3(
    idata,
    var_names: Sequence[str] = ("delta_pf", "delta_ft", "contrast_pf_ft"),
    convergence: Optional[dict] = None,
    title: str = "M3 convergence diagnostics",
):
    """Trace + rank plots for the key M3 parameters with a convergence banner.

    Parameters
    ----------
    idata : az.InferenceData
    var_names : sequence of str
        Scalar parameters to diagnose.
    convergence : dict, optional
        ``{"max_rhat": .., "min_ess_bulk": .., "n_divergent": ..}`` for the
        annotation (from ``results.json['m3_convergence']``).
    """
    import arviz as az

    present = [v for v in var_names if v in idata.posterior.data_vars]
    n = len(present)
    fig, axes = plt.subplots(n, 2, figsize=(9.5, 2.1 * n + 1.0), squeeze=False)

    # Left column: trace (per chain), drawn explicitly for layout control.
    # Right column: ArviZ rank plot.
    for r, v in enumerate(present):
        da = idata.posterior[v]
        chains = da.coords["chain"].values
        # Trace.
        for ch in chains:
            axes[r, 0].plot(
                da.sel(chain=ch).values,
                linewidth=0.6,
                alpha=0.8,
                label=f"chain {int(ch)}",
            )
        axes[r, 0].set_ylabel(v)
        if r == 0:
            axes[r, 0].set_title("Trace (4 chains)")
        if r == n - 1:
            axes[r, 0].set_xlabel("Draw")
        # Rank plot (ArviZ) on the right axis.
        try:
            az.plot_rank(idata, var_names=[v], ax=axes[r, 1], kind="bars")
            axes[r, 1].set_title("Rank plot" if r == 0 else "")
        except Exception:  # noqa: BLE001  fall back to ESS-less marker
            axes[r, 1].text(
                0.5,
                0.5,
                "rank plot unavailable",
                ha="center",
                va="center",
                transform=axes[r, 1].transAxes,
            )
        axes[r, 1].set_ylabel("")

    sup = title
    if convergence is not None:
        sup += (
            f"   (R̂ max = {convergence.get('max_rhat', float('nan')):.3f},  "
            f"ESS_bulk min = {convergence.get('min_ess_bulk', float('nan')):.0f},  "
            f"divergences = {int(convergence.get('n_divergent', 0))})"
        )
    fig.suptitle(sup, y=1.005, fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    return fig


# --- fig7 : inter-surgical delay ECDF + fitted distribution ---------------


def delay_ecdf_fit(
    df_wide: pd.DataFrame,
    idata=None,
    value_col: str = "inter_surgery_d",
    group_col: str = "group",
    medians: Optional[dict] = None,
    title: str = "Inter-surgical delay by group (H4)",
):
    """Empirical ECDF of the inter-surgical delay + fitted LogNormal per group.

    The fitted curve uses the posterior-mean ``mu``/``sigma`` of the M5
    LogNormal (``idata_m5.nc``) when ``idata`` is provided; otherwise it falls
    back to a per-group MLE LogNormal fit. Medians are annotated.

    Parameters
    ----------
    df_wide : pd.DataFrame
    idata : az.InferenceData, optional
        M5 posterior with ``mu``/``sigma`` indexed by ``group`` (log scale).
    value_col : str
    medians : dict, optional
        ``{group: median_days}``; if None, computed empirically.
    """
    from scipy import stats as _st

    sub = df_wide[[group_col, value_col]].dropna().copy()
    groups = [g for g in GROUP_ORDER if g in sub[group_col].unique()]

    # Posterior-mean log-scale params from M5 idata, if available.
    post_params: dict = {}
    if idata is not None and "mu" in getattr(idata, "posterior", {}).data_vars:
        mu_da = idata.posterior["mu"]
        sig_da = idata.posterior["sigma"]
        gcoords = list(map(str, mu_da.coords["group"].values))
        for g in groups:
            if g in gcoords:
                post_params[g] = (
                    float(mu_da.sel(group=g).values.mean()),
                    float(sig_da.sel(group=g).values.mean()),
                )

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    xmax = float(sub[value_col].max()) * 1.05
    grid = np.linspace(1, xmax, 400)
    for grp in groups:
        vals = np.sort(sub.loc[sub[group_col] == grp, value_col].astype(float).values)
        color = _group_color(grp)
        # Empirical ECDF (step).
        ecdf_y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(
            vals,
            ecdf_y,
            where="post",
            color=color,
            linewidth=2.0,
            label=f"{GROUP_LABELS.get(grp, grp)} (empirical, n={len(vals)})",
        )
        # Fitted LogNormal CDF.
        if grp in post_params:
            mu, sigma = post_params[grp]
            fit_src = "M5 posterior"
        else:
            logv = np.log(vals[vals > 0])
            mu, sigma = float(logv.mean()), float(logv.std(ddof=1))
            fit_src = "MLE"
        cdf = _st.norm.cdf((np.log(grid) - mu) / sigma)
        ax.plot(
            grid,
            cdf,
            color=color,
            linestyle="--",
            linewidth=1.6,
            alpha=0.9,
            label=f"{GROUP_LABELS.get(grp, grp)} LogNormal ({fit_src})",
        )
        # Median guide.
        med = (medians or {}).get(grp)
        if med is None:
            med = float(np.median(vals))
        ax.axvline(med, color=color, linestyle=":", linewidth=1.2, alpha=0.8)
        ax.annotate(
            f"median {med:.0f} d",
            (med, 0.5),
            xytext=(6, 0),
            textcoords="offset points",
            rotation=90,
            va="center",
            ha="left",
            fontsize=9,
            color=color,
        )

    ax.set_xlabel("Inter-surgical delay (days)")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, xmax)
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.5)
    ax.text(
        0.02,
        0.97,
        "Mediator (downstream of group), not a confounder:\n"
        "cyclops are re-operated sooner  adjusting it strengthens the PF effect.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#bbb"),
    )
    fig.tight_layout()
    return fig


# --- figS1 (supplementary) : PF slopegraph S1→S2 by group -----------------


def slopegraph_pf_mpl(
    df_wide: pd.DataFrame,
    value_s1: str = "lesion_pf_S1",
    value_s2: str = "lesion_pf_S2",
    group_col: str = "group",
    title: str = "Patellofemoral score S1 → S2 by group",
):
    """Matplotlib paired slopegraph of the PF score S1→S2, faceted by group.

    Supplementary publication figure (static PNG counterpart of the Plotly
    :func:`slopegraph_paired_annotated`). Each patient is a thin line; the
    group median trajectory is overlaid in bold; the worsening % is annotated.
    """
    groups = [g for g in GROUP_ORDER if g in df_wide[group_col].unique()]
    rng = np.random.default_rng(RANDOM_SEED)
    fig, axes = plt.subplots(1, len(groups), figsize=(8.4, 4.8), sharey=True)
    if len(groups) == 1:
        axes = [axes]

    for ax, grp in zip(axes, groups):
        sub = df_wide[df_wide[group_col] == grp].dropna(subset=[value_s1, value_s2])
        s1 = sub[value_s1].astype(float).values
        s2 = sub[value_s2].astype(float).values
        color = _group_color(grp)
        jit = rng.normal(0, 0.02, size=len(sub))
        for a, b, j in zip(s1, s2, jit):
            ax.plot([0 + j, 1 + j], [a, b], color=color, alpha=0.4, linewidth=1.0)
        # Median trajectory.
        ax.plot(
            [0, 1],
            [np.median(s1), np.median(s2)],
            color="black",
            linewidth=3.0,
            marker="D",
            markersize=8,
            zorder=4,
            label="median",
        )
        deltas = s2 - s1
        pct_worse = float((deltas > 0).mean() * 100.0)
        ax.set_title(
            f"{GROUP_LABELS.get(grp, grp)}\nworsened {pct_worse:.0f}% (n={len(sub)})"
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["S1", "S2"])
        ax.set_xlim(-0.3, 1.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Patellofemoral lesion score (trochlea + patella)")
    fig.suptitle(title, y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# --- fig10 : flexum  the mechanical driver -------------------------------


def flexum_panel(
    df_flexum: pd.DataFrame,
    spearman_rho: Optional[float] = None,
    spearman_ci: Optional[Sequence[float]] = None,
    spearman_p: Optional[float] = None,
    fisher_p: Optional[float] = None,
    group_col: str = "group",
    flexum_col: str = "flexum_pre_s2",
    delta_col: str = "delta_lesion_pf",
):
    """Two-panel flexum figure: group separation + intra-cyclops dose-response.

    (a) Pre-S2 extension deficit by group  cyclops carry a flexum (−3…−10°),
    meniscus sit at 0° (no nodule), a near-total separation (Fisher p tiny).
    (b) Within cyclops, the flexum *depth* vs the patellofemoral progression
    Δ_PF, with the Spearman ρ: a present/absent **mechanistic marker**, not a
    graded dose (ρ ≈ 0). The honest pair  strong group signal, no within-group
    dose-response.
    """
    order = [g for g in ("meniscus", "cyclops") if g in df_flexum[group_col].unique()]
    pal = {g: _group_color(g) for g in order}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # (a) deficit by group  box + strip.
    sns.boxplot(
        data=df_flexum,
        x=group_col,
        y=flexum_col,
        order=order,
        hue=group_col,
        palette=pal,
        legend=False,
        width=0.5,
        showcaps=False,
        fliersize=0,
        boxprops=dict(alpha=0.35),
        ax=axa,
    )
    sns.stripplot(
        data=df_flexum,
        x=group_col,
        y=flexum_col,
        order=order,
        hue=group_col,
        palette=pal,
        legend=False,
        jitter=0.22,
        size=5,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.4,
        ax=axa,
    )
    axa.axhline(0, color="#222", lw=0.9, ls="--")
    axa.set_xlabel("")
    axa.set_ylabel("Pre-S2 extension deficit (°)  flexum")
    axa.set_xticklabels([GROUP_LABELS.get(g, g) for g in order])
    axa.set_title("(a) Flexum by group")
    if fisher_p is not None:
        axa.text(
            0.5,
            0.03,
            f"any flexum: Fisher p = {fisher_p:.1e}",
            transform=axa.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4f9f0", ec="#9fcb86"),
        )

    # (b) dose-response within cyclops: deficit (positive degrees) vs Δ_PF.
    cyc = df_flexum[
        (df_flexum[group_col] == "cyclops") & df_flexum[delta_col].notna()
    ].copy()
    cyc["deficit"] = -cyc[flexum_col].astype(float)
    rng = np.random.default_rng(RANDOM_SEED)
    jx = cyc["deficit"].values + rng.uniform(-0.18, 0.18, len(cyc))
    jy = cyc[delta_col].astype(float).values + rng.uniform(-0.06, 0.06, len(cyc))
    axb.scatter(
        jx,
        jy,
        s=46,
        color=_group_color("cyclops"),
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    axb.axhline(0, color="#222", lw=0.8, ls=":")
    axb.set_xlabel("Flexum depth (° of extension lost)")
    axb.set_ylabel("Δ patellofemoral lesion score (S2 − S1)")
    axb.set_title("(b) Dose-response within cyclops")
    if spearman_rho is not None:
        lab = f"Spearman ρ = {spearman_rho:+.2f}"
        if spearman_ci is not None:
            lab += f"  [{spearman_ci[0]:+.2f}, {spearman_ci[1]:+.2f}]"
        if spearman_p is not None:
            verdict = "no graded dose-response" if spearman_p > 0.05 else "graded"
            lab += f"\np = {spearman_p:.2f} → {verdict}"
        axb.text(
            0.03,
            0.97,
            lab,
            transform=axb.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff3e9", ec="#fc8d62"),
        )

    fig.suptitle(
        "Flexum  the mechanical driver (present in cyclops, absent in meniscus)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --- fig0a / fig0b : first-glance descriptive views -----------------------


def descriptive_demographics(patient: pd.DataFrame, group_col: str = "group"):
    """First-glance demographic panel by group: age, % female, BMI.

    Three small multiples (violin + strip for age and BMI, a proportion bar for
    sex) so the reader sees the cohort imbalance the Love plot quantifies.
    """
    order = [g for g in ("meniscus", "cyclops") if g in patient[group_col].unique()]
    pal = {g: _group_color(g) for g in order}
    p = patient.copy()
    for c in ("age_at_trauma", "imc", "female"):
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")

    fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.2))
    # (a) age
    sns.violinplot(
        data=p,
        x=group_col,
        y="age_at_trauma",
        order=order,
        hue=group_col,
        palette=pal,
        legend=False,
        inner="quartile",
        cut=0,
        ax=axs[0],
    )
    sns.stripplot(
        data=p,
        x=group_col,
        y="age_at_trauma",
        order=order,
        color="#333",
        size=3,
        alpha=0.5,
        jitter=0.18,
        ax=axs[0],
    )
    axs[0].set_title("(a) Âge au trauma")
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Âge (ans)")
    axs[0].set_xticklabels([GROUP_LABELS.get(g, g) for g in order])
    # (b) % female
    fem = p.groupby(group_col)["female"].mean().reindex(order) * 100.0
    axs[1].bar(
        range(len(order)),
        fem.values,
        color=[pal[g] for g in order],
        width=0.6,
        edgecolor="white",
    )
    for i, v in enumerate(fem.values):
        axs[1].text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=10)
    axs[1].set_xticks(range(len(order)))
    axs[1].set_xticklabels([GROUP_LABELS.get(g, g) for g in order])
    axs[1].set_ylim(0, 100)
    axs[1].set_ylabel("% féminin")
    axs[1].set_title("(b) Sexe (% féminin)")
    # (c) BMI
    sns.violinplot(
        data=p,
        x=group_col,
        y="imc",
        order=order,
        hue=group_col,
        palette=pal,
        legend=False,
        inner="quartile",
        cut=0,
        ax=axs[2],
    )
    sns.stripplot(
        data=p,
        x=group_col,
        y="imc",
        order=order,
        color="#333",
        size=3,
        alpha=0.5,
        jitter=0.18,
        ax=axs[2],
    )
    axs[2].set_title("(c) IMC")
    axs[2].set_xlabel("")
    axs[2].set_ylabel("IMC (kg/m²)")
    axs[2].set_xticklabels([GROUP_LABELS.get(g, g) for g in order])

    fig.suptitle(
        "Démographie de base par groupe (cyclops vs méniscus)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def descriptive_lesion_baseline(wide: pd.DataFrame, group_col: str = "group"):
    """First-glance baseline cartilage view by group at S1.

    (a) Total S1 lesion load (sum of 6 compartments) by group  box + strip;
    (b) per-compartment prevalence of any lesion (score ≥ 1) at S1 by group
    grouped bars. Shows the starting cartilage state the equivalence test
    (fig1b) summarises.
    """
    order = [g for g in ("meniscus", "cyclops") if g in wide[group_col].unique()]
    pal = {g: _group_color(g) for g in order}
    w = wide.copy()
    w["lesion_total_S1"] = pd.to_numeric(w["lesion_total_S1"], errors="coerce")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.8, 4.4))
    # (a) total S1 load
    sns.boxplot(
        data=w,
        x=group_col,
        y="lesion_total_S1",
        order=order,
        hue=group_col,
        palette=pal,
        legend=False,
        width=0.5,
        showcaps=False,
        fliersize=0,
        boxprops=dict(alpha=0.35),
        ax=axa,
    )
    sns.stripplot(
        data=w,
        x=group_col,
        y="lesion_total_S1",
        order=order,
        hue=group_col,
        palette=pal,
        legend=False,
        jitter=0.2,
        size=4,
        alpha=0.7,
        ax=axa,
    )
    axa.set_title("(a) Charge lésionnelle S1 (somme 6 compartiments)")
    axa.set_xlabel("")
    axa.set_ylabel("Score lésionnel total S1")
    axa.set_xticklabels([GROUP_LABELS.get(g, g) for g in order])
    # (b) per-compartment S1 prevalence (% with score ≥ 1)
    labels, cyc_v, men_v = [], [], []
    for s in SITES:
        col = f"{s}_S1"
        if col not in w.columns:
            continue
        labels.append(s)
        for g, store in (("cyclops", cyc_v), ("meniscus", men_v)):
            sub = pd.to_numeric(w.loc[w[group_col] == g, col], errors="coerce").dropna()
            store.append(100.0 * float((sub >= 1).mean()) if len(sub) else 0.0)
    x = np.arange(len(labels))
    bw = 0.38
    axb.bar(
        x - bw / 2,
        men_v,
        bw,
        label=GROUP_LABELS.get("meniscus", "meniscus"),
        color=_group_color("meniscus"),
        edgecolor="white",
    )
    axb.bar(
        x + bw / 2,
        cyc_v,
        bw,
        label=GROUP_LABELS.get("cyclops", "cyclops"),
        color=_group_color("cyclops"),
        edgecolor="white",
    )
    axb.set_xticks(x)
    axb.set_xticklabels(labels, rotation=20, ha="right")
    axb.set_ylabel("% avec lésion ≥ 1 à S1")
    axb.set_title("(b) Prévalence lésionnelle par compartiment (S1)")
    axb.legend(fontsize=8)

    fig.suptitle(
        "État cartilagineux de départ (S1) par groupe", fontsize=12, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
