import matplotlib.pyplot as plt
import seaborn as sns
import statannotations.Annotator

import rstats


def boxplot(df, x, y):
    sns.set_style("whitegrid")
    sns.boxplot(data=df, x=x, y=y, fill=False, color="#4c4c4c", linewidth=2.0)
    sns.stripplot(data=df, x=x, y=y, jitter=True, color="#4c4c4c", alpha=0.5)
    sns.despine()
    plt.xticks(rotation=90)


def boxplot_glmm(df, metric, title=None, verbose=False):
    """Make a boxplot of raw data by `category`, overlay model-predicted marginal means
    with error bars, and annotate Tukey pairwise significance (GLMM fit in R via rpy2).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing at least [metric, category, id].
    metric : str
        Column name of the response variable.
    title : str, optional
        Plot title. Defaults to "GLMM estimates & raw spread".
    verbose : bool, default=False
        If True, verbose output from statannotations.

    Returns
    -------
    res : R model object (lmerMod/glmmTMB)
    pred : pandas.DataFrame
        Marginal means per category (from emmeans).
    pairs : pandas.DataFrame
        Tukey pairwise contrasts with p-values.
    """
    # 1) Fit model in R
    formula = f"{metric} ~ category + (1|id)"
    res = rstats.fit_glmm(df, formula, family="gaussian")

    # 2) Marginal means + Tukey
    pred = rstats.marginal_means_by_category(res, term="category")
    pairs = rstats.pairwise_tukey(res, term="category")

    # Split contrast string into group1/group2
    if "contrast" in pairs.columns:
        pairs[["group1", "group2"]] = pairs["contrast"].str.split(" - ", expand=True)

    # Categories for plotting
    cats = sorted(df["category"].dropna().unique())
    xmap = {c: i for i, c in enumerate(cats)}

    # Pair tuples ordered by span (short → long)
    pair_tuples = [(g1, g2) for g1, g2 in zip(pairs["group1"], pairs["group2"])]
    pair_tuples = sorted(pair_tuples, key=lambda p: abs(xmap[p[1]] - xmap[p[0]]))
    pvalues = [
        pairs.loc[(pairs.group1 == a) & (pairs.group2 == b), "p.value"].iloc[0]
        for (a, b) in pair_tuples
    ]

    # --- Plotting ---
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()

    # Boxplot
    sns.boxplot(
        data=df,
        x="category",
        y=metric,
        order=cats,
        color="#4c4c4c",
        fill=False,
        showcaps=True,
        width=0.5,
        linewidth=2.0,
        ax=ax,
    )

    # Jittered raw points
    sns.stripplot(
        data=df,
        x="category",
        y=metric,
        order=cats,
        color="#4c4c4c",
        alpha=0.45,
        jitter=True,
        size=4,
        ax=ax,
    )

    # Model estimates (emm) ± SE
    ax.errorbar(
        pred["category"],
        pred["emmean"],
        yerr=pred["SE"],
        fmt="o",
        color="black",
        capsize=3,
    )

    # Significance stars
    if pair_tuples:
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
            text_format="star", hide_non_significant=True, loc="inside", color="#4c4c4c"
        )
        annotator.set_pvalues(pvalues)
        annotator.annotate()

    # Aesthetics
    sns.despine()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.set_title(title or "GLMM estimates & raw spread")
    ax.set_xlabel("Category")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.5)
    plt.tight_layout()

    return res, pred, pairs
