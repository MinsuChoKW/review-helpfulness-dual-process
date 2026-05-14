

suppressPackageStartupMessages({
  library(dplyr)
  library(glmmTMB)
})

REPO    <- "/Users/minsucho/Documents/Helpfulness/revisions"
LATENT  <- file.path(REPO, "data", "latent_data")
OUT_CSV <- file.path(REPO, "results", "robustness", "C4_exposed_subsample.csv")
dir.create(dirname(OUT_CSV), showWarnings = FALSE, recursive = TRUE)

PLATFORMS <- c("amazon", "audible", "coursera", "hotel")

fit_one <- function(platform) {
  cat("\n=== ", platform, " ===\n", sep = "")
  df <- read.csv(file.path(LATENT, paste0(platform, ".csv")), stringsAsFactors = FALSE)
  n_full <- nrow(df)
  df_pos <- df %>% filter(Helpfulness > 0)
  n_pos  <- nrow(df_pos)
  cat(sprintf("  full N: %d   subsample N (Helpfulness > 0): %d   retention: %.4f\n",
              n_full, n_pos, n_pos / n_full))

  fit_random_slopes <- NULL
  fit_random_intercept <- NULL
  used_fallback <- FALSE
  err_msg <- ""

  res <- tryCatch({
    fit_random_slopes <- glmmTMB(
      Helpfulness ~ Latent_Heuristic + Latent_Systematic
                  + (Latent_Heuristic + Latent_Systematic | Group),
      family = nbinom2,
      data   = df_pos
    )
    list(model = fit_random_slopes, fallback = FALSE, err = "")
  }, error = function(e) {
    cat("  random-slopes model failed:", conditionMessage(e), "\n")
    cat("  falling back to random-intercept model\n")
    fit_random_intercept <- glmmTMB(
      Helpfulness ~ Latent_Heuristic + Latent_Systematic + (1 | Group),
      family = nbinom2,
      data   = df_pos
    )
    list(model = fit_random_intercept, fallback = TRUE, err = conditionMessage(e))
  })

  model <- res$model
  used_fallback <- res$fallback
  err_msg <- res$err

  if (is.null(model)) stop("both models failed for ", platform)

  # Convergence check
  conv <- !is.null(model$sdr$pdHess) && model$sdr$pdHess
  cat("  convergence (PD Hessian):", conv, "\n")

  cs <- summary(model)$coefficients$cond
  # Pull fixed-effect rows for the two predictors of interest
  rows <- list()
  for (term in c("Latent_Heuristic", "Latent_Systematic")) {
    if (term %in% rownames(cs)) {
      est <- cs[term, "Estimate"]
      se  <- cs[term, "Std. Error"]
      z   <- cs[term, "z value"]
      pv  <- cs[term, "Pr(>|z|)"]
      rows[[term]] <- c(est = est, se = se, z = z, p = pv)
    } else {
      rows[[term]] <- c(est = NA, se = NA, z = NA, p = NA)
    }
  }
  print(cs)

  data.frame(
    platform = platform,
    n_full = n_full,
    n_pos  = n_pos,
    retention = round(n_pos / n_full, 4),
    used_fallback_random_intercept = as.integer(used_fallback),
    convergence_pdHess = as.integer(conv),
    fallback_reason = err_msg,
    sys_estimate = round(rows$Latent_Systematic["est"], 4),
    sys_std_error = round(rows$Latent_Systematic["se"], 4),
    sys_z = round(rows$Latent_Systematic["z"], 4),
    sys_p_value = round(rows$Latent_Systematic["p"], 4),
    heu_estimate = round(rows$Latent_Heuristic["est"], 4),
    heu_std_error = round(rows$Latent_Heuristic["se"], 4),
    heu_z = round(rows$Latent_Heuristic["z"], 4),
    heu_p_value = round(rows$Latent_Heuristic["p"], 4),
    stringsAsFactors = FALSE
  )
}

stars <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  "n.s."
}

main <- function() {
  parts <- list()
  for (p in PLATFORMS) {
    parts[[p]] <- fit_one(p)
  }
  full <- do.call(rbind, parts)
  full$sys_sig <- vapply(full$sys_p_value, stars, character(1))
  full$heu_sig <- vapply(full$heu_p_value, stars, character(1))
  write.csv(full, OUT_CSV, row.names = FALSE)
  cat("\nWrote", OUT_CSV, "(", nrow(full), "rows )\n")
  print(full, row.names = FALSE)
}

if (!interactive()) main()
