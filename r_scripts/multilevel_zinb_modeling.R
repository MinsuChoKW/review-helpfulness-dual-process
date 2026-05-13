# Baseline Multilevel Zero-Inflated Negative Binomial regression (manuscript Table 6).
#
# Purpose
#   Reproduce the multilevel ZINB coefficients reported in Table 6 by fitting,
#   for each platform, a glmmTMB::nbinom2 model with random slopes for
#   Latent_Heuristic and Latent_Systematic by Group (item-level).
#   Both the count and the zero-inflation components carry the same fixed
#   and random structure.
#
# Inputs
#   ../data/latent_data/<platform>.csv  (same files as zinb_modeling.R)
#
# Outputs
#   Prints per-platform summary(), AIC, BIC, marginal/conditional R^2
#   (MuMIn::r.squaredGLMM), and McFadden pseudo-R^2 to stdout.
#
# Manuscript section: Section VI.B (results) and Table 6.
#
# Note on convergence: on some platforms the random-slopes ZINB has a
# non-positive-definite Hessian. The robustness driver
# r_scripts/robustness/run_all_zinb.R falls back to a random-intercept-only
# model when this happens; in the baseline tabulation we report the
# random-slopes fit and flag any convergence issues in the manuscript text.

library(dplyr)
library(plspm)
library(glmmTMB)
library(MuMIn)

base_path <- "../data/latent_data/"

datasets <- list(
  "amazon"   = read.csv(paste0(base_path, "amazon.csv")),
  "audible"  = read.csv(paste0(base_path, "audible.csv")),
  "coursera" = read.csv(paste0(base_path, "coursera.csv")),
  "hotel"    = read.csv(paste0(base_path, "hotel.csv"))
)

for (platform in names(datasets)) {
  cat("\n===============================\n")
  cat("### Running:", platform, "###\n")

  df <- datasets[[platform]]

  zinb_model <- glmmTMB(
    Helpfulness ~ Latent_Heuristic + Latent_Systematic
                + (Latent_Heuristic + Latent_Systematic | Group),
    ziformula = ~ Latent_Heuristic + Latent_Systematic
                + (Latent_Heuristic + Latent_Systematic | Group),
    family = nbinom2,
    data = df
  )

  print(summary(zinb_model))

  AIC_value <- AIC(zinb_model)
  BIC_value <- BIC(zinb_model)
  cat("AIC:", AIC_value, "\n")
  cat("BIC:", BIC_value, "\n")

  r2 <- r.squaredGLMM(zinb_model)
  print(r2)

  null_model <- update(zinb_model, . ~ 1)
  llh <- logLik(zinb_model)
  llhNull <- logLik(null_model)
  mcfadden <- 1 - as.numeric(llh / llhNull)
  cat("McFadden R²:", round(mcfadden, 4), "\n")
}
