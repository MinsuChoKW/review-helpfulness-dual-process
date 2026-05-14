
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
