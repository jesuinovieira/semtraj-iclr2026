import matplotlib.pyplot as plt
import seaborn as sns
import statannotations.Annotator

import stats


def boxplot(df, x, y):
    sns.set_style("whitegrid")
    sns.boxplot(data=df, x=x, y=y, fill=False, color="#4c4c4c", linewidth=2.0)
    sns.stripplot(data=df, x=x, y=y, jitter=True, color="#4c4c4c", alpha=0.5)
    sns.despine()
    plt.xticks(rotation=90)


def boxplot_with_model(df, metric, title=None, verbose=False):
    """
    - seaborn boxplot + jittered points (single fixed color)
    - black dot + errorbar at model-estimated marginal mean (fixed effects)
    - significance stars from Tukey via statannotations
    """
    # 1) Fit model and compute predicted marginal means + CIs
    res = stats.fit_lmm_category(df, metric)
    pred = stats.emmeans_like_predictions(res, df, metric)

    # 2) Tukey pairwise on raw data  (expects columns: group1, group2, p.value)
    pairs = stats.tukey_pairwise(df, metric)
    cats = sorted(df["category"].dropna().unique())
    xmap = {c: i for i, c in enumerate(cats)}

    # Sort pairs by span (short → long) so stacking is clean
    pair_tuples = [(g1, g2) for g1, g2 in zip(pairs["group1"], pairs["group2"])]
    pair_tuples = sorted(pair_tuples, key=lambda p: abs(xmap[p[1]] - xmap[p[0]]))
    pvalues = [
        pairs.loc[(pairs.group1 == a) & (pairs.group2 == b), "p.value"].iloc[0]
        for (a, b) in pair_tuples
    ]

    # 3) Plot
    sns.set_style("whitegrid")
    _, ax = plt.subplots()

    # Boxplot
    sns.boxplot(
        data=df,
        x="category",
        y=metric,
        order=cats,
        color="#4c4c4c",
        fill=False,
        showcaps=True,
        ax=ax,
        width=0.5,
        linewidth=2.0,
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

    # Model estimates + CI
    xmap = {c: i for i, c in enumerate(cats)}
    for _, r in pred.iterrows():
        x = xmap[r["x"]]
        ax.errorbar(
            x,
            r["predicted"],
            yerr=[[r["predicted"] - r["conf.low"]], [r["conf.high"] - r["predicted"]]],
            fmt="o",
            elinewidth=2,
            capsize=6,
            color="#4c4c4c",
            zorder=5,
        )

    # Significance stars via statannotations (use our Tukey p-values)
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
        # NOTE: set pvalue_format to configure significance codes
        annotator.configure(text_format="star", loc="inside", color="#4c4c4c")
        annotator.set_pvalues(pvalues)
        annotator.annotate()

    sns.despine()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_title(title or "GLMM estimates & raw spread")
    ax.set_xlabel("category")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.5)
    plt.tight_layout()

    return res, pred, pairs
