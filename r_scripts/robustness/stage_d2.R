

suppressPackageStartupMessages({
  library(dplyr)
  library(glmmTMB)
})

REPO    <- "/Users/minsucho/Documents/Helpfulness/revisions"
LATENT  <- file.path(REPO, "data", "latent_data")
OUT_CSV <- file.path(REPO, "results", "robustness", "D2_item_FE.csv")
dir.create(dirname(OUT_CSV), showWarnings = FALSE, recursive = TRUE)

PLATFORMS <- c("amazon", "audible", "coursera", "hotel")

extract_term <- function(coef_mat, term) {
  if (term %in% rownames(coef_mat)) {
    c(est = coef_mat[term, "Estimate"],
      se  = coef_mat[term, "Std. Error"],
      z   = coef_mat[term, "z value"],
      p   = coef_mat[term, "Pr(>|z|)"])
  } else {
    c(est = NA, se = NA, z = NA, p = NA)
  }
}

stars <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  "n.s."
}

fit_one <- function(platform) {
  cat("\n=== ", platform, " ===\n", sep = "")
  df <- read.csv(file.path(LATENT, paste0(platform, ".csv")), stringsAsFactors = FALSE)
  n <- nrow(df)
  cat("  N:", n, "  unique Groups:", length(unique(df$Group)), "\n")

  df <- df %>%
    group_by(Group) %>%
    mutate(
      Latent_Heuristic_wc  = Latent_Heuristic  - mean(Latent_Heuristic),
      Latent_Systematic_wc = Latent_Systematic - mean(Latent_Systematic)
    ) %>%
    ungroup()

  used_fallback <- FALSE
  err_msg <- ""

  # Try random-slopes first
  fit <- tryCatch({
    glmmTMB(
      Helpfulness ~ Latent_Heuristic_wc + Latent_Systematic_wc
                  + (Latent_Heuristic_wc + Latent_Systematic_wc | Group),
      ziformula = ~ Latent_Heuristic_wc + Latent_Systematic_wc
                  + (Latent_Heuristic_wc + Latent_Systematic_wc | Group),
      family = nbinom2,
      data   = df
    )
  }, error = function(e) {
    err_msg <<- paste("random-slopes errored:", conditionMessage(e))
    cat("  random-slopes model errored:", conditionMessage(e), "\n")
    NULL
  })

  # Check convergence (PD Hessian). If not converged, fall back to RI-only.
  conv_ok <- !is.null(fit) && !is.null(fit$sdr$pdHess) && fit$sdr$pdHess
  if (!conv_ok) {
    cat("  random-slopes did not converge (PD Hessian = ", conv_ok, "); falling back to random-intercept-only\n", sep = "")
    err_msg <- if (nchar(err_msg) == 0) "random-slopes Hessian non-PD" else err_msg
    fit <- tryCatch({
      glmmTMB(
        Helpfulness ~ Latent_Heuristic_wc + Latent_Systematic_wc + (1 | Group),
        ziformula = ~ Latent_Heuristic_wc + Latent_Systematic_wc + (1 | Group),
        family = nbinom2,
        data   = df
      )
    }, error = function(e) {
      cat("  random-intercept-only model also errored:", conditionMessage(e), "\n")
      err_msg <<- paste(err_msg, "; RI-only errored:", conditionMessage(e))
      NULL
    })
    used_fallback <- TRUE
  }

  if (is.null(fit)) stop("both D2 model variants failed for ", platform)

  conv <- !is.null(fit$sdr$pdHess) && fit$sdr$pdHess
  cat("  convergence (PD Hessian):", conv, "\n")

  s <- summary(fit)
  cs_cond <- s$coefficients$cond
  cs_zi   <- s$coefficients$zi

  sys_c <- extract_term(cs_cond, "Latent_Systematic_wc")
  heu_c <- extract_term(cs_cond, "Latent_Heuristic_wc")
  sys_z <- extract_term(cs_zi,   "Latent_Systematic_wc")
  heu_z <- extract_term(cs_zi,   "Latent_Heuristic_wc")

  cat("  conditional fixed effects:\n"); print(cs_cond)
  cat("  zero-inflation fixed effects:\n"); print(cs_zi)

  data.frame(
    platform = platform,
    n = n,
    used_fallback_random_intercept = as.integer(used_fallback),
    convergence_pdHess = as.integer(conv),
    fallback_reason = err_msg,
    sys_cond_estimate = round(sys_c["est"], 4),
    sys_cond_se = round(sys_c["se"], 4),
    sys_cond_z = round(sys_c["z"], 4),
    sys_cond_p = round(sys_c["p"], 4),
    sys_cond_sig = stars(sys_c["p"]),
    heu_cond_estimate = round(heu_c["est"], 4),
    heu_cond_se = round(heu_c["se"], 4),
    heu_cond_z = round(heu_c["z"], 4),
    heu_cond_p = round(heu_c["p"], 4),
    heu_cond_sig = stars(heu_c["p"]),
    sys_zi_estimate = round(sys_z["est"], 4),
    sys_zi_se = round(sys_z["se"], 4),
    sys_zi_z = round(sys_z["z"], 4),
    sys_zi_p = round(sys_z["p"], 4),
    sys_zi_sig = stars(sys_z["p"]),
    heu_zi_estimate = round(heu_z["est"], 4),
    heu_zi_se = round(heu_z["se"], 4),
    heu_zi_z = round(heu_z["z"], 4),
    heu_zi_p = round(heu_z["p"], 4),
    heu_zi_sig = stars(heu_z["p"]),
    stringsAsFactors = FALSE
  )
}

main <- function() {
  parts <- list()
  for (p in PLATFORMS) {
    parts[[p]] <- fit_one(p)
  }
  full <- do.call(rbind, parts)
  write.csv(full, OUT_CSV, row.names = FALSE)
  cat("\nWrote", OUT_CSV, "(", nrow(full), "rows )\n")
  print(full, row.names = FALSE)
}

if (!interactive()) main()
