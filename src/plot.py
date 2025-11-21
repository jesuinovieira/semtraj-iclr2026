import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# FIXME: must have latex installed in this path, maybe try and catch error
os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]
plt.rcParams["figure.dpi"] = 300

base = 12
plt.rcParams.update(
    {
        "font.size": base,
        "axes.titlesize": base + 3,
        "axes.labelsize": base + 2,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "figure.titlesize": base + 5,
    }
)


def boxplot(df, metric, pred, pairs, cats, ax=None, figsize=(6, 6)):
    # Establish a consistent categorical order
    xmap = {c: i for i, c in enumerate(cats)}

    # Ensure both the raw df and the EMM table use the same categorical order
    df = df.copy()
    df["category"] = pd.Categorical(df["category"], categories=cats, ordered=True)
    if "category" in pred:
        pred = pred.copy()
        pred["category"] = pd.Categorical(
            pred["category"], categories=cats, ordered=True
        )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    palette = sns.color_palette("colorblind", n_colors=len(cats))
    # palette = sns.color_palette("Set2", n_colors=len(cats))

    # Boxplot of raw data (same order as cats)
    # Color the lines instead
    sns.boxplot(
        data=df,
        x="category",
        y=metric,
        order=cats,
        color="#4c4c4c",
        fill=False,
        palette=palette,
        hue="category",
        showcaps=False,
        width=0.5,
        linewidth=1.75,
        showfliers=False,
        ax=ax,
        zorder=1,
    )

    # Jittered raw points
    sns.stripplot(
        data=df,
        x="category",
        y=metric,
        order=cats,
        # color="#4c4c4c",
        palette=palette,
        hue="category",
        alpha=0.15,
        jitter=True,
        size=5,
        ax=ax,
        zorder=1,
    )

    # Model estimates (EMMs) ± SE
    xpos = [xmap[c] for c in pred["category"]]
    ax.errorbar(
        xpos,
        pred["emmean"],
        yerr=pred["SE"],
        fmt="o",
        color="black",
        capsize=5,
        capthick=1.5,
        markersize=5,
        linewidth=1.5,
        zorder=3,  # Above raw points
    )

    # 5) Aesthetics
    sns.despine(top=True, right=True, left=True, bottom=True)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, axis="y", alpha=0.5)
    ax.tick_params(axis="both", which="major")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    return ax


def heatmap(pairs, cats, ax):
    # Build a (symmetric) matrix of p-values
    pmat = pd.DataFrame(1.0, index=cats, columns=cats, dtype=float)
    for _, row in pairs.iterrows():
        g1, g2 = str(row["group1"]), str(row["group2"])
        p = float(row["p.value"])
        pmat.loc[g1, g2] = p
        pmat.loc[g2, g1] = p

    np.fill_diagonal(pmat.values, np.nan)
    mask = np.tril(np.ones_like(pmat, dtype=bool))

    colors = sns.light_palette("seagreen", n_colors=5, reverse=True)
    colors[-1] = (1, 1, 1)
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 1e-4, 1e-3, 1e-2, 5e-2, 1.0]  # last one is max
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.grid(False)
    hm = sns.heatmap(
        pmat.astype(float),
        mask=mask,
        # TODO: annot directly here with stars?
        # annot=pmat.map(stars).to_numpy(),
        cmap=cmap,
        norm=norm,
        cbar=False,
        vmin=0,
        vmax=1,
        xticklabels=cats,
        yticklabels=cats,
        ax=ax,
        square=True,
    )

    ax.tick_params(axis="both", which="major")
    ax.tick_params(axis="x", rotation=90)

    return hm, pmat
