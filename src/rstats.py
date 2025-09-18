import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import default_converter
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

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
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


def _r_df_to_py(r_df):
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_df)


def glmm(df: pd.DataFrame, formula: str, family: str = "lognormal"):
    """Fit a GLMM with glmmTMB via rpy2.

    For log links (default), the response must be strictly positive; we do not modify
    data here. Clean your data (e.g., add a tiny epsilon) upstream if needed.
    """
    ro.globalenv["df_py"] = _py_df_to_r(df)
    ro.globalenv["formula_str"] = formula
    ro.globalenv["family"] = family

    ro.r(
        r"""
        if (!requireNamespace("glmmTMB", quietly = TRUE)) {
          stop("Package 'glmmTMB' is not installed. Run install.packages('glmmTMB').")
        }

        df <- df_py
        formula <- as.formula(formula_str)
        fam <- family

        # Map our string -> family object
        famobj <- switch(fam,
          "gamma"     = stats::Gamma(link = "log"),
          "poisson"   = stats::poisson(link = "log"),
          "binomial"  = stats::binomial(),
          "lognormal" = stats::gaussian(link = "log"),  # lognormal via log link
          "gaussian"  = stats::gaussian(),
          stop(paste("Unsupported family:", fam))
        )

        # NOTE: glmmTMB does not auto-convert character IDs to factors.
        # If your RHS has (1|id) and id is character/integer, glmmTMB will coerce,
        # but it's safer to ensure factors in Python beforehand.

        mod <- glmmTMB::glmmTMB(
          formula = formula,
          data    = df,
          family  = famobj
        )

        cat("GLMM fitted with glmmTMB\n", strrep("-", 70), "\n\n", sep = "")
        print(summary(mod))

        # Return via the global env so rpy2 can fetch it
        assign("mod", mod, envir = .GlobalEnv)
        """
    )
    return ro.globalenv["mod"]


def emmeans(mod, term: str = "category") -> pd.DataFrame:
    ro.globalenv["mod"] = mod
    ro.globalenv["term_str"] = term

    r_df = ro.r(
        r"""
        if (!requireNamespace("emmeans", quietly = TRUE)) {
          stop("Package 'emmeans' is not installed. Run install.packages('emmeans').")
        }

        term <- term_str

        # FIXME: discuss type = "response" (use or not, affects scale on plots). It's
        # currently being computed twice, here and in function below (double check it)

        # type = "response" puts results on the natural scale, with bias correction where relevant
        emm <- emmeans::emmeans(mod, specs = as.formula(paste("~", term)), type = "response")
        out <- as.data.frame(emm)

        # Normalize column names: emmeans on response scale yields 'response' not 'emmean'
        if ("response" %in% names(out)) names(out)[names(out) == "response"] <- "emmean"

        # SE is usually 'SE' already; keep a defensive rename in case a method returns 'SE.df'
        if (!("SE" %in% names(out)) && ("SE.df" %in% names(out))) {
          names(out)[names(out) == "SE.df"] <- "SE"
        }

        # Standardize the term column name to the term string (e.g., 'category')
        # emmeans typically uses the factor name already, so this is just a sanity pass
        names(out)[names(out) == names(out)[1]] <- term

        out
        """
    )
    return _r_df_to_py(r_df)


def pairs(mod, term: str = "category") -> pd.DataFrame:
    """Tukey-adjusted pairwise comparisons for 'term'.

    By default, p-values are computed on the linear predictor.
    """
    ro.globalenv["mod"] = mod
    ro.globalenv["term_str"] = term

    r_df = ro.r(
        r"""
        if (!requireNamespace("emmeans", quietly = TRUE)) {
          stop("Package 'emmeans' is not installed. Run install.packages('emmeans').")
        }

        term <- term_str
        emm  <- emmeans::emmeans(mod, specs = as.formula(paste("~", term)))
        pw   <- pairs(emm, adjust = "tukey")

        cat("\nTukey pairwise comparisons\n")
        cat(strrep("-", 70), "\n\n")
        print(pw)
        as.data.frame(pw)
        """
    )
    return _r_df_to_py(r_df)
