import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import default_converter
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Ensure needed R packages are installed (quiet, no spam)
ro.r('options(repos="https://cloud.r-project.org")')
ro.r(
    """
ensure_pkgs <- function(pkgs) {
  to_get <- pkgs[!pkgs %in% rownames(installed.packages())]
  if (length(to_get)) {
    suppressMessages(suppressWarnings(
      install.packages(to_get, quiet=TRUE)
    ))
  }
}
"""
)
ro.r('ensure_pkgs(c("glmmTMB","emmeans","DHARMa","pbkrtest","lmerTest","broom.mixed"))')

# Now load them
ro.r(
    "library(glmmTMB); "
    "library(emmeans); "
    "library(DHARMa); "
    "library(pbkrtest); "
    "library(lmerTest); "
    "library(broom.mixed)"
)


def _py_df_to_r(df: pd.DataFrame):
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


def _r_df_to_py(r_df):
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_df)


def fit_glmm(df: pd.DataFrame, formula: str, family: str = "lognormal"):
    """Fit a GLMM with glmmTMB via rpy2.

    Notes
    -----
    - Default family is "lognormal" implemented as gaussian(link="log").
    - For log links, the response must be strictly positive; we do not modify data here.
      Prefer to pre-clean your data (e.g., add a tiny epsilon) upstream if needed.
    """
    ro.globalenv["df_py"] = _py_df_to_r(df)
    ro.globalenv["formula_str"] = formula
    ro.globalenv["fam_str"] = family

    ro.r(
        r"""
        if (!requireNamespace("glmmTMB", quietly = TRUE)) {
          stop("Package 'glmmTMB' is not installed. Run install.packages('glmmTMB').")
        }

        df  <- df_py
        fml <- as.formula(formula_str)
        fam <- fam_str

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
          formula = fml,
          data    = df,
          family  = famobj
        )

        print(summary(mod))

        # Return via the global env so rpy2 can fetch it
        assign("mod", mod, envir = .GlobalEnv)
        """
    )
    return ro.globalenv["mod"]


def marginal_means_by_category(mod, term: str = "category") -> pd.DataFrame:
    """Get marginal means (EMMs) by 'term' on the RESPONSE scale (back-transformed).

    Returns columns 'category' (or your term), 'emmean', 'SE', 'lower.CL', 'upper.CL'.
    """
    ro.globalenv["mod"] = mod
    ro.globalenv["term_str"] = term

    r_df = ro.r(
        r"""
        if (!requireNamespace("emmeans", quietly = TRUE)) {
          stop("Package 'emmeans' is not installed. Run install.packages('emmeans').")
        }

        term <- term_str

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


def pairwise_tukey(mod, term: str = "category") -> pd.DataFrame:
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
        as.data.frame(pw)
        """
    )
    return _r_df_to_py(r_df)


def fixed_effects_table(mod, conf_level: float = 0.95) -> pd.DataFrame:
    """Return fixed-effect coefficients on the *link* scale with SE, z/t, p, and CIs."""
    ro.globalenv["mod"] = mod
    ro.globalenv["conf_level"] = conf_level
    r_df = ro.r(
        r"""
        if (!requireNamespace("broom.mixed", quietly = TRUE)) {
          stop("Package 'broom.mixed' is not installed. Run install.packages('broom.mixed').")
        }

        out <- broom.mixed::tidy(
          mod,
          effects = "fixed",
          conf.int = TRUE,
          conf.level = conf_level
        )

        # Keep only the usual columns and standardize names a bit
        out <- out[, c("term","estimate","std.error","statistic","p.value","conf.low","conf.high")]
        names(out) <- c("term","estimate","SE","stat","p.value","conf.low","conf.high")
        out
        """
    )
    return _r_df_to_py(r_df)


def random_effects_var(mod, conf_level: float = 0.95) -> pd.DataFrame:
    """Return random-effect variance parameters (SD/Var, correlations if present)."""
    ro.globalenv["mod"] = mod
    ro.globalenv["conf_level"] = conf_level
    r_df = ro.r(
        r"""
        if (!requireNamespace("broom.mixed", quietly = TRUE)) {
          stop("Package 'broom.mixed' is not installed. Run install.packages('broom.mixed').")
        }

        out <- broom.mixed::tidy(mod, effects = "ran_pars", conf.int = TRUE, conf.level = conf_level)
        out
        """
    )
    return _r_df_to_py(r_df)
