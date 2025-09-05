import formulaic
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.multicomp


def mean_sd_by_group(df, group, metric):
    """Return mean and sd per group (like rstatix::get_summary_stats)."""
    g = df.groupby(group)[metric]
    data = {
        f"{group}": g.mean().index,
        "mean": g.mean().values,
        "sd": g.std(ddof=1).values,
        "n": g.size().values,
    }
    out = pd.DataFrame(data)
    return out.sort_values(group).reset_index(drop=True)


def fit_lmm(df, metric):
    """Fits a linear mixed model with category as fixed effect and concept as random
    effect: metric ~ C(category) + (1|id) + (1|concept)
    """
    # Keep only necessary columns, drop rows with NA on any of them
    cols = ["id", "category", "concept", metric]
    df = df.loc[:, cols].dropna().reset_index(drop=True).copy()

    # MixedLM: one primary grouping (id) + variance component per concept
    md = sm.MixedLM.from_formula(
        f"{metric} ~ C(category)",
        data=df,
        groups=df["id"],
        re_formula="1",
        vc_formula={"concept": "0 + C(concept)"},  # Random intercept per concept
    )
    res = md.fit(method="lbfgs", maxiter=200, disp=False)
    return res


def marginal_means_by_category(res, d, metric):
    """Compute model-estimated marginal means (like emmeans in R)."""
    # All category levels used in the model
    levels = list(pd.Categorical(d["category"]).categories)

    # Column order expected by the fitted model
    design_cols = list(res.model.exog_names)

    rows = []
    for c in levels:
        row = pd.DataFrame({"category": pd.Categorical([c], categories=levels)})

        # Build model matrix with explicit levels; include intercept explicitly.
        # Using repr(levels) ensures the literal list is embedded in the formula.
        formula = f"1 + C(category, levels={repr(levels)})"
        X = formulaic.model_matrix(formula, row, output="pandas")

        # Align columns to model’s exog (add missing, order match)
        X = X.reindex(columns=design_cols, fill_value=0.0)

        beta = res.fe_params.reindex(design_cols).values
        pred = float(X.values @ beta)

        V = res.cov_params().loc[design_cols, design_cols].values
        se = float(np.sqrt(X.values @ V @ X.values.T))
        rows.append(
            {
                "x": c,
                "predicted": pred,
                "conf.low": pred - 1.96 * se,
                "conf.high": pred + 1.96 * se,
            }
        )

    return pd.DataFrame(rows)


def tukey_pairwise(d, metric):
    ret_cols = ["group1", "group2", "p.value", "meandiff", "lower", "upper"]

    # Minimal cleaning so Tukey doesn't return NaNs
    tmp = d[["category", metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["category", metric])
    tmp["category"] = tmp["category"].astype(str)

    # Drop groups with < 2 points (common cause of NaN p-values)
    cnt = tmp.groupby("category")[metric].size()
    keep = cnt[cnt >= 2].index
    tmp = tmp[tmp["category"].isin(keep)]

    # Nothing to compare
    if tmp["category"].nunique() < 2:
        return pd.DataFrame(columns=ret_cols)

    mc = statsmodels.stats.multicomp.MultiComparison(tmp[metric], tmp["category"])
    tk = mc.tukeyhsd()

    summary_df = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])

    # Convert to numeric
    for col in ["meandiff", "p-adj", "lower", "upper"]:
        summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    # Align with your R-style expectations
    summary_df = summary_df.rename(columns={"p-adj": "p.value"})

    # Guard against any lingering NaNs
    summary_df["p.value"] = summary_df["p.value"].fillna(1.0)

    return summary_df[ret_cols]
