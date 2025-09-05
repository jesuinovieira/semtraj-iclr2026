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
ro.r('ensure_pkgs(c("glmmTMB","lme4","emmeans","DHARMa","pbkrtest","lmerTest"))')

# Now load them
ro.r(
    "library(glmmTMB); library(lme4); library(emmeans); library(DHARMa); "
    "library(pbkrtest); library(lmerTest)"
)


def _py_df_to_r(df: pd.DataFrame):
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


def _r_df_to_py(r_df):
    with localconverter(default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_df)


def fit_glmm(df: pd.DataFrame, formula: str, family: str = "gaussian"):
    ro.globalenv["df_py"] = _py_df_to_r(df)
    ro.globalenv["formula_str"] = formula
    ro.globalenv["fam_str"] = family

    ro.r(
        """
        df <- df_py
        fml <- as.formula(formula_str)
        fam <- fam_str

        if (identical(fam, "gaussian")) {
          mod <- lme4::lmer(fml, data = df)
        } else {
          famobj <- switch(fam,
            "gamma"     = stats::Gamma(link="log"),
            "poisson"   = stats::poisson(link="log"),
            "binomial"  = stats::binomial(),
            "lognormal" = stats::gaussian(link="log"),
            stats::gaussian()
          )
          mod <- glmmTMB::glmmTMB(formula = fml, data = df, family = famobj)
        }
    """
    )
    return ro.globalenv["mod"]


def marginal_means_by_category(mod, term: str = "category") -> pd.DataFrame:
    ro.globalenv["mod"] = mod
    ro.globalenv["term_str"] = term
    r_df = ro.r(
        """
        fml_specs <- as.formula(paste0("~", term_str))
        as.data.frame(emmeans::emmeans(mod, specs = fml_specs))
    """
    )
    return _r_df_to_py(r_df)


def pairwise_tukey(mod, term: str = "category") -> pd.DataFrame:
    ro.globalenv["mod"] = mod
    ro.globalenv["term_str"] = term
    r_df = ro.r(
        """
        emm <- emmeans::emmeans(mod, specs = term_str)
        as.data.frame(pairs(emm, adjust = "tukey"))
    """
    )
    return _r_df_to_py(r_df)
