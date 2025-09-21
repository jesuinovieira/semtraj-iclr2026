import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import conversion as cv
from rpy2.robjects import pandas2ri

# Ensure needed R packages are installed (quiet, no spam)
ro.r(
    """
    options(repos="https://cloud.r-project.org")
    ensure_pkgs <- function(pkgs) {
      to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
      if (length(to_install)) {
        suppressMessages(suppressWarnings(install.packages(to_install, quiet=TRUE)))
      }
    }
    ensure_pkgs(c("glmmTMB","emmeans"))
    suppressPackageStartupMessages(library(glmmTMB))
    suppressPackageStartupMessages(library(emmeans))
    """
)


def _py_df_to_r(df: pd.DataFrame):
    with cv.localconverter(cv.get_conversion() + pandas2ri.converter):
        return cv.py2rpy(df)


def _r_df_to_py(obj):
    with cv.localconverter(cv.get_conversion() + pandas2ri.converter):
        return cv.rpy2py(obj)


def glmm(df: pd.DataFrame, formula: str, family: str = "lognormal"):
    """Fit a GLMM with glmmTMB via rpy2.

    For log links (default), the response must be strictly positive; we do not modify
    data here. Clean your data (e.g., add a tiny epsilon) upstream if needed.
    """
    ro.globalenv["df_py"] = _py_df_to_r(df)
    ro.globalenv["formula_str"] = formula
    ro.globalenv["family"] = family

    model = ro.r(
        r"""
        df <- df_py
        formula <- as.formula(formula_str)
        fam <- family

        famobj <- switch(fam,
          "lognormal" = glmmTMB::lognormal(),
          "gaussian" = stats::gaussian(),
          stop(paste("Unsupported family:", fam))
        )

        # Fit the model
        glmm <- glmmTMB::glmmTMB(formula = formula, data = df, family = famobj)

        cat("GLMM fitted with glmmTMB\n", strrep("-", 70), "\n\n", sep = "")
        print(summary(glmm))

        glmm
        """
    )
    return model


def emmeans(effect: str = "category") -> pd.DataFrame:
    ro.globalenv["effect_str"] = effect
    r_df = ro.r(
        r"""
        # Sanity check
        if (!exists("glmm", envir = .GlobalEnv)) stop("`glmm` not found")

        # Get estimated marginal means
        effect <- effect_str
        emm <- emmeans::emmeans(glmm, specs = as.formula(paste("~", effect)))

        # Keep emm on link but materialize output on response
        out <- as.data.frame(emm, type = "response")
        out
        """
    )
    return _r_df_to_py(r_df)


def pairs() -> pd.DataFrame:
    r_df = ro.r(
        r"""
        # Sanity check
        if (!exists("emm", envir = .GlobalEnv)) stop("`emm` not found")

        # Pairwise comparisons with Tukey adjustment
        pw <- pairs(emm, adjust = "tukey")

        cat("\nTukey pairwise comparisons (link scale)\n")
        cat(strrep("-", 70), "\n\n")
        print(pw)
        as.data.frame(pw)
        """
    )
    return _r_df_to_py(r_df)
