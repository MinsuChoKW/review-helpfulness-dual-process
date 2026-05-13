# Capture baseline Standard ZINB (Table 5) and Multilevel ZINB (Table 6)
# fixed-effect coefficients into a single CSV that figures/figure3_zinb_comparison.py
# consumes. Run once; figure script reads the CSV.
#
# Inputs
#   ../data/latent_data/<platform>.csv
#
# Output
#   ../results/baseline_zinb_coefficients.csv
#     columns: platform, model, component, construct, estimate, se, p_value, sig_marker
#     model in {standard, multilevel}; component in {conditional, zero_inflation};
#     construct in {Systematic, Heuristic}.
#
# Manuscript section: Tables 5 and 6 (Section VI.A and VI.B).

suppressPackageStartupMessages({
  library(glmmTMB)
})

REPO    <- "/Users/minsucho/Documents/Helpfulness/revisions"
LATENT  <- file.path(REPO, "data", "latent_data")
OUT_CSV <- file.path(REPO, "results", "baseline_zinb_coefficients.csv")
dir.create(dirname(OUT_CSV), showWarnings = FALSE, recursive = TRUE)

PLATFORMS <- c("amazon", "audible", "coursera", "hotel")

stars <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  "n.s."
}

extract_rows <- function(fit, platform, model_label) {
  s <- summary(fit)
  cs_cond <- s$coefficients$cond
  cs_zi   <- s$coefficients$zi
  rows <- list()
  for (comp in c("conditional", "zero_inflation")) {
    cs <- if (comp == "conditional") cs_cond else cs_zi
    for (construct in c("Systematic", "Heuristic")) {
      term <- paste0("Latent_", construct)
      if (term %in% rownames(cs)) {
        est <- cs[term, "Estimate"]
        se  <- cs[term, "Std. Error"]
        p   <- cs[term, "Pr(>|z|)"]
      } else {
        est <- NA; se <- NA; p <- NA
      }
      rows[[length(rows) + 1]] <- data.frame(
        platform   = platform,
        model      = model_label,
        component  = comp,
        construct  = construct,
        estimate   = round(est, 4),
        se         = round(se, 4),
        p_value    = round(p, 6),
        sig_marker = stars(p),
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

all_rows <- list()
for (platform in PLATFORMS) {
  cat(sprintf("\n=== %s ===\n", platform))
  df <- read.csv(file.path(LATENT, paste0(platform, ".csv")))

  cat("  fitting standard ZINB ...\n")
  std <- glmmTMB(
    Helpfulness ~ Latent_Heuristic + Latent_Systematic,
    ziformula = ~ Latent_Heuristic + Latent_Systematic,
    family = nbinom2, data = df
  )
  all_rows[[length(all_rows) + 1]] <- extract_rows(std, platform, "standard")

  cat("  fitting multilevel ZINB ...\n")
  ml <- tryCatch({
    glmmTMB(
      Helpfulness ~ Latent_Heuristic + Latent_Systematic
                  + (Latent_Heuristic + Latent_Systematic | Group),
      ziformula = ~ Latent_Heuristic + Latent_Systematic
                  + (Latent_Heuristic + Latent_Systematic | Group),
      family = nbinom2, data = df
    )
  }, error = function(e) {
    cat("    random-slopes errored, falling back to random-intercept-only:",
        conditionMessage(e), "\n")
    NULL
  })
  conv <- !is.null(ml) && !is.null(ml$sdr$pdHess) && ml$sdr$pdHess
  if (!conv) {
    cat("    random-slopes non-PD Hessian; falling back to random-intercept-only\n")
    ml <- glmmTMB(
      Helpfulness ~ Latent_Heuristic + Latent_Systematic + (1 | Group),
      ziformula = ~ Latent_Heuristic + Latent_Systematic + (1 | Group),
      family = nbinom2, data = df
    )
  }
  all_rows[[length(all_rows) + 1]] <- extract_rows(ml, platform, "multilevel")
}

full <- do.call(rbind, all_rows)
write.csv(full, OUT_CSV, row.names = FALSE)
cat(sprintf("\nWrote %s (%d rows)\n", OUT_CSV, nrow(full)))
print(full, row.names = FALSE)
