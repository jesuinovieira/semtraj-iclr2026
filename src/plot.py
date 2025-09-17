import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statannotations.Annotator

import rstats


def boxplot(df, x, y):
    sns.set_style("whitegrid")
    sns.boxplot(data=df, x=x, y=y, fill=False, color="#4c4c4c", linewidth=2.0)
    sns.stripplot(data=df, x=x, y=y, jitter=True, color="#4c4c4c", alpha=0.5)
    sns.despine()
    plt.xticks(rotation=90)


def boxplot_glmm(df, metric, family="lognormal", title=None, verbose=False):
    """Boxplot of raw data + GLMM marginal means (±SE) and Tukey star annotations.

    Assumes:
      - rstats.fit_glmm uses glmmTMB
      - rstats.marginal_means_by_category returns response-scale EMMs with columns:
        ['category', 'emmean', 'SE', 'lower.CL', 'upper.CL', ...]
      - rstats.pairwise_tukey returns a data frame with 'contrast' and 'p.value'
    """
    # 1) Fit model in R (GLMM)
    formula = f"{metric} ~ category + (1|id)"
    res = rstats.fit_glmm(df, formula=formula, family=family)

    # 2) Marginal means (on RESPONSE scale) + Tukey pairwise tests
    pred = rstats.marginal_means_by_category(res, term="category")
    pairs = rstats.pairwise_tukey(res, term="category")

    # Defensive normalize just in case helpers changed; keep plotting code stable
    if "emmean" not in pred and "response" in pred:
        pred = pred.rename(columns={"response": "emmean"})
    if "SE" not in pred and "SE.df" in pred:
        pred = pred.rename(columns={"SE.df": "SE"})

    # Split "A - B" contrasts into separate columns for statannotations
    if "contrast" in pairs.columns and not {"group1", "group2"}.issubset(pairs.columns):
        pairs[["group1", "group2"]] = pairs["contrast"].str.split(" - ", expand=True)

    # 3) Establish a consistent categorical order
    # Sort categories alphabetically for stability, or pass a custom order if you have one
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
    fig, ax = plt.subplots()

    # Boxplot of raw data (same order as cats)
    sns.boxplot(
        data=df,
        x="category",
        y=metric,
        order=cats,
        color="#4c4c4c",
        fill=False,
        showcaps=True,
        width=0.5,
        linewidth=1.75,
        showfliers=False,
        ax=ax,
    )

    # Jittered raw points (helps visualize within-group spread)
    sns.stripplot(
        data=df,
        x="category",
        y=metric,
        order=cats,
        color="#4c4c4c",
        alpha=0.4,
        jitter=True,
        size=3,
        ax=ax,
        zorder=1,
    )

    # Model estimates (EMMs) ± SE (on RESPONSE scale!)
    xpos = [xmap[c] for c in pred["category"]]
    ax.errorbar(
        xpos,
        pred["emmean"],
        yerr=pred["SE"],
        fmt="o",
        color="black",
        capsize=5,
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
            loc="inside",
            color="#4c4c4c",
        )
        annotator.set_pvalues(pvalues)
        annotator.annotate()

    # 5) Aesthetics

    sns.despine()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.set_title(title or "GLMM estimates & raw spread")
    ax.set_xlabel("Category")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.5)
    plt.tight_layout()

    return res, pred, pairs
