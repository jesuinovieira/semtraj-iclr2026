import contextlib
import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statannotations.Annotator

import rstats


def boxplot(df, metric, family="lognormal", title=None, verbose=False):
    """Boxplot of raw data + GLMM marginal means (±SE) and Tukey star annotations.

    Assumes:
      - rstats.fit_glmm uses glmmTMB
      - rstats.marginal_means_by_category returns response-scale EMMs with columns:
        ['category', 'emmean', 'SE', 'lower.CL', 'upper.CL', ...]
      - rstats.pairwise_tukey returns a data frame with 'contrast' and 'p.value'
    """
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        # 1) Fit GLMM with random intercepts for 'id' and fixed effect of 'category'
        formula = f"{metric} ~ category + (1|id)"
        res = rstats.glmm(df, formula=formula, family=family)

        # 2) Get marginal means and pairwise comparisons
        pred = rstats.emmeans(effect="category")
        pairs = rstats.pairs()

    # Standardize column names for EMMs
    if "emmean" not in pred and "response" in pred:
        pred = pred.rename(columns={"response": "emmean"})
    if "SE" not in pred and "SE.df" in pred:
        pred = pred.rename(columns={"SE.df": "SE"})

    # Split "A - B" contrasts into separate columns for statannotations
    if "contrast" in pairs.columns and not {"group1", "group2"}.issubset(pairs.columns):
        pairs[["group1", "group2"]] = pairs["contrast"].str.split(" - ", expand=True)

    # 3) Establish a consistent categorical order
    cats = sorted(pd.Series(df["category"]).dropna().unique())
    xmap = {c: i for i, c in enumerate(cats)}

    # Ensure both the raw df and the EMM table use the same categorical order
    df = df.copy()
    df["category"] = pd.Categorical(df["category"], categories=cats, ordered=True)
    if "category" in pred:
        pred = pred.copy()
        pred["category"] = pd.Categorical(
            pred["category"], categories=cats, ordered=True
        )

    # Build pair tuples in increasing span so brackets don't overlap too much
    pair_tuples = []
    if {"group1", "group2"}.issubset(pairs.columns):
        pair_tuples = [(g1, g2) for g1, g2 in zip(pairs["group1"], pairs["group2"])]
        pair_tuples = sorted(pair_tuples, key=lambda p: abs(xmap[p[1]] - xmap[p[0]]))

    # Extract p-values in the same order as pair_tuples
    pvalues = []
    if len(pair_tuples) and "p.value" in pairs.columns:
        pvalues = [
            pairs.loc[(pairs.group1 == a) & (pairs.group2 == b), "p.value"].iloc[0]
            for (a, b) in pair_tuples
            if not pairs.loc[(pairs.group1 == a) & (pairs.group2 == b), "p.value"].empty
        ]

    # 4) Plot
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    fig, ax = plt.subplots()

    # Boxplot of raw data (same order as cats)
    sns.boxplot(
        data=df,
        x="category",
        y=metric,
        order=cats,
        color="#4c4c4c",
        fill=False,
        # palette=palette,
        # hue="category",
        showcaps=True,
        width=0.5,
        linewidth=1.25,
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
        color="#4c4c4c",
        alpha=0.25,
        jitter=True,
        size=3,
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

    # Significance stars from Tukey-adjusted comparisons
    if pair_tuples and len(pvalues) == len(pair_tuples):
        annotator = statannotations.Annotator.Annotator(
            ax,
            pair_tuples,
            data=df,
            x="category",
            y=metric,
            order=cats,
            verbose=verbose,
        )
        annotator.configure(
            text_format="star",
            hide_non_significant=True,
            loc="outside",
            color="black",
            line_width=0.75,
            fontsize=10,
        )
        annotator.set_pvalues(pvalues)
        annotator.annotate()

    # 5) Aesthetics
    sns.despine(top=True, right=True, left=True, bottom=True)
    ax.set_xlabel("")
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, axis="y", alpha=0.5)
    plt.tight_layout()

    return res, pred, pairs, stdout.getvalue()
