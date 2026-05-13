# Baseline Standard Zero-Inflated Negative Binomial regression (manuscript Table 5).
#
# Purpose
#   Reproduce the standard ZINB coefficients reported in Table 5 of the
#   manuscript by fitting, for each of the four platforms (Amazon, Audible,
#   Coursera, Booking.com / hotel), a glmmTMB::nbinom2 model with no
#   random effects. Both the count and the zero-inflation components are
#   linear in Latent_Heuristic and Latent_Systematic.
#
# Inputs
#   ../data/latent_data/<platform>.csv     for platform in {amazon, audible,
#                                            coursera, hotel}
#       columns: Helpfulness, Group, Latent_Helpfulness, Latent_Heuristic,
#                Latent_Systematic
#
# Outputs
#   Prints per-platform summary(), AIC, BIC, marginal/conditional R^2
#   (MuMIn::r.squaredGLMM), and McFadden pseudo-R^2 to stdout.
#   Coefficients are also captured by figures/figure3_zinb_comparison.py
#   via a parallel batch run; this script itself does not write CSVs.
#
# Manuscript section: Section VI.A (results) and Table 5.

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
    Helpfulness ~ Latent_Heuristic + Latent_Systematic,
    ziformula = ~ Latent_Heuristic + Latent_Systematic,
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
