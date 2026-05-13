#!/usr/bin/env Rscript
# Stage 0 re-verification using the R plspm package (CRAN version 0.6.0,
# installed via micromamba into ~/.local/r-env).
#
# Engine swap only: indicator features (data/robustness/features_baseline/*.csv)
# are reused verbatim from the Python-side Stage 0 run. The PLS-SEM
# specification is the same as the Python script:
#   - Heuristic  (mode B): rating_deviation, title_length, recency
#                          (coursera drops title_length)
#   - Systematic (mode B): depth, breadth, readability, arousal
#   - Outcome    (mode A, single MV): Helpfulness
#   - Inner scheme: "path"; bootstrap: br = 1000
#   - Scaling: pre-z-standardize each column, then call plspm(..., scaled=FALSE)
#     to match what we did on the Python side.
#
# Outputs:
#   results/robustness/stage0_plspm_R_partial_<platform>.csv  (per-platform)
#   results/robustness/stage0_plspm_R_vs_smartpls.csv         (combined)

suppressPackageStartupMessages({
  library(plspm)
})

REPO     <- "/Users/minsucho/Documents/Helpfulness/revisions"
FEAT_DIR <- file.path(REPO, "data", "robustness", "features_baseline")
OUT_DIR  <- file.path(REPO, "results", "robustness")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

PLATFORMS <- c("amazon", "audible", "coursera", "hotel")
SYS_INDS  <- c("depth", "breadth", "readability", "arousal")
HEU_FULL  <- c("rating_deviation", "title_length", "recency")
HEU_COURS <- c("rating_deviation", "recency")

# SmartPLS 4 reference values from manuscript Tables 2 and 3.
smartpls_ref <- list(
  amazon   = list(depth=1.026, breadth=-0.085, readability=0.044, arousal=-0.026,
                  rating_deviation=0.574, title_length=0.736, recency=0.125),
  hotel    = list(depth=1.040, breadth=0.251,  readability=0.012, arousal=0.085,
                  rating_deviation=0.919, title_length=0.390, recency=0.139),
  audible  = list(depth=1.028, breadth=-0.108, readability=0.048, arousal=-0.034,
                  rating_deviation=0.582, title_length=0.142, recency=0.755),
  coursera = list(depth=1.024, breadth=-0.159, readability=0.042, arousal=-0.063,
                  rating_deviation=0.986, recency=0.227)
)
smartpls_sig <- list(
  amazon   = list(depth="***", breadth="***", readability="***", arousal="***",
                  rating_deviation="***", title_length="***", recency="n.s."),
  hotel    = list(depth="***", breadth="***", readability="n.s.", arousal="***",
                  rating_deviation="***", title_length="***", recency="***"),
  audible  = list(depth="***", breadth="***", readability="***", arousal="n.s.",
                  rating_deviation="***", title_length="**",  recency="***"),
  coursera = list(depth="***", breadth="***", readability="**", arousal="***",
                  rating_deviation="***", recency="**")
)

# Pre-standardize each column (mean 0, sd 1 with ddof=1) before plspm — the
# same workaround used in Python because the metric scaling there pooled
# across columns rather than per-column. plspm R's scaled=TRUE does work
# correctly per-column, but we keep pre-standardization for byte-equivalent
# inputs between the two engines.
zscore_cols <- function(df, cols) {
  for (c in cols) {
    v <- as.numeric(df[[c]])
    df[[c]] <- (v - mean(v)) / sd(v)
  }
  df
}

stars_from_p <- function(p) {
  if (is.na(p) || p > 1) return("")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  "n.s."
}

run_platform <- function(platform) {
  cat("\n=== ", platform, " ===\n", sep = "")
  feat_path <- file.path(FEAT_DIR, paste0(platform, ".csv"))
  df <- read.csv(feat_path, stringsAsFactors = FALSE)
  heu <- if (platform == "coursera") HEU_COURS else HEU_FULL
  needed <- c("Helpfulness", heu, SYS_INDS)
  df <- df[, needed]
  n_pre <- nrow(df)
  df <- df[complete.cases(df), ]
  n_post <- nrow(df)
  cat("  rows:", n_pre, "->", n_post, "after dropna\n")
  df <- zscore_cols(df, needed)

  # Inner model (path matrix): lower triangular, target rows = endogenous.
  # Order of LVs: Heuristic, Systematic, Outcome. Outcome has incoming paths
  # from both Heuristic and Systematic.
  lvs <- c("Heuristic", "Systematic", "Outcome")
  path <- matrix(0L, nrow = 3, ncol = 3, dimnames = list(lvs, lvs))
  path["Outcome", "Heuristic"]  <- 1L
  path["Outcome", "Systematic"] <- 1L

  blocks <- list(
    Heuristic  = heu,
    Systematic = SYS_INDS,
    Outcome    = c("Helpfulness")
  )
  modes <- c("B", "B", "A")

  t0 <- Sys.time()
  fit <- plspm(
    Data        = df,
    path_matrix = path,
    blocks      = blocks,
    modes       = modes,
    scheme      = "path",
    scaled      = FALSE,         # data is already pre-z-scored
    boot.val    = TRUE,
    br          = 1000,
    dataset     = TRUE
  )
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  plspm fit: %.1fs\n", dt))

  outer  <- fit$outer_model     # data.frame: name, block, weight, loading, communality, redundancy
  boot_w <- fit$boot$weights    # data.frame: Original, Mean.Boot, Std.Error, perc.025, perc.975
  paths  <- fit$path_coefs      # square matrix

  cat("  outer_model:\n")
  outer_print <- outer
  for (col in c("weight", "loading", "communality", "redundancy")) {
    outer_print[[col]] <- round(outer_print[[col]], 4)
  }
  print(outer_print)
  cat("  bootstrap weights:\n"); print(round(boot_w, 4))
  cat("  path coefficients:\n"); print(round(paths, 4))

  rows <- data.frame(
    platform = character(0), construct = character(0), indicator = character(0),
    n = integer(0),
    plspm_R_weight = numeric(0), plspm_R_boot_mean = numeric(0),
    plspm_R_boot_se = numeric(0), plspm_R_t_stat = numeric(0),
    plspm_R_ci_lo = numeric(0), plspm_R_ci_hi = numeric(0),
    plspm_R_p_value = numeric(0), plspm_R_sig = character(0),
    smartpls_weight = numeric(0), smartpls_sig = character(0),
    abs_diff = numeric(0), sign_match = integer(0), sig_match = integer(0),
    ci_contains_smartpls = integer(0),
    stringsAsFactors = FALSE
  )

  for (block_name in c("Heuristic", "Systematic")) {
    inds <- blocks[[block_name]]
    for (ind in inds) {
      w  <- outer$weight[outer$name == ind & outer$block == block_name]
      bw <- boot_w[rownames(boot_w) == paste0(block_name, "-", ind), , drop = FALSE]
      if (nrow(bw) == 0) {
        bw <- boot_w[rownames(boot_w) == ind, , drop = FALSE]
      }
      if (nrow(bw) == 0) {
        # Bootstrap rows in plspm 0.6 are indexed by paste(block, name, sep="-") or just name; try original column lookup
        next
      }
      orig  <- bw[1, "Original"]
      mbo   <- bw[1, "Mean.Boot"]
      se    <- bw[1, "Std.Error"]
      lo    <- bw[1, "perc.025"]
      hi    <- bw[1, "perc.975"]
      tstat <- if (is.finite(se) && se > 0) orig / se else NA_real_
      p_val <- if (is.finite(tstat)) 2 * (1 - pnorm(abs(tstat))) else NA_real_
      sigp  <- stars_from_p(p_val)

      ref      <- if (!is.null(smartpls_ref[[platform]][[ind]])) smartpls_ref[[platform]][[ind]] else NA_real_
      ref_sig  <- if (!is.null(smartpls_sig[[platform]][[ind]])) smartpls_sig[[platform]][[ind]] else ""
      abs_d    <- if (!is.na(ref)) abs(w - ref) else NA_real_
      sgn      <- if (!is.na(ref)) as.integer(sign(w) == sign(ref) || (w == 0 && ref == 0)) else NA_integer_
      sigm     <- if (!is.na(ref) && nchar(sigp) > 0 && nchar(ref_sig) > 0) as.integer(sigp == ref_sig) else NA_integer_
      contains <- if (!is.na(ref) && is.finite(lo) && is.finite(hi)) as.integer(lo <= ref && ref <= hi) else NA_integer_

      rows <- rbind(rows, data.frame(
        platform = platform, construct = block_name, indicator = ind,
        n = n_post,
        plspm_R_weight = round(w, 4), plspm_R_boot_mean = round(mbo, 4),
        plspm_R_boot_se = round(se, 4), plspm_R_t_stat = round(tstat, 4),
        plspm_R_ci_lo = round(lo, 4), plspm_R_ci_hi = round(hi, 4),
        plspm_R_p_value = round(p_val, 4), plspm_R_sig = sigp,
        smartpls_weight = ref, smartpls_sig = ref_sig,
        abs_diff = round(abs_d, 4), sign_match = sgn, sig_match = sigm,
        ci_contains_smartpls = contains,
        stringsAsFactors = FALSE
      ))
    }
  }
  out_path <- file.path(OUT_DIR, paste0("stage0_plspm_R_partial_", platform, ".csv"))
  write.csv(rows, out_path, row.names = FALSE)
  cat("  wrote", out_path, "\n")
  rows
}

main <- function() {
  parts <- list()
  for (p in PLATFORMS) {
    parts[[p]] <- run_platform(p)
  }
  full <- do.call(rbind, parts)
  out_csv <- file.path(OUT_DIR, "stage0_plspm_R_vs_smartpls.csv")
  write.csv(full, out_csv, row.names = FALSE)
  cat("\nWrote", out_csv, "(", nrow(full), "rows )\n")

  # Quick summary
  cat("\n--- R plspm success-criteria assessment ---\n")
  cat(sprintf("  cells total:                       %d\n", nrow(full)))
  cat(sprintf("  |plspm_R - smartpls| <= 0.03:      %d/%d\n",
              sum(full$abs_diff <= 0.03, na.rm = TRUE), nrow(full)))
  cat(sprintf("  sign match:                        %d/%d\n",
              sum(full$sign_match == 1, na.rm = TRUE), nrow(full)))
  cat(sprintf("  significance category match:       %d/%d (threshold: 25)\n",
              sum(full$sig_match == 1, na.rm = TRUE), nrow(full)))
  cat(sprintf("  bootstrap CI contains smartpls:    %d/%d\n",
              sum(full$ci_contains_smartpls == 1, na.rm = TRUE), nrow(full)))

  cat("\n--- breadth side-by-side (R plspm) ---\n")
  b <- full[full$indicator == "breadth", c("platform", "plspm_R_weight",
                                            "plspm_R_boot_se", "plspm_R_ci_lo",
                                            "plspm_R_ci_hi", "smartpls_weight",
                                            "ci_contains_smartpls")]
  print(b, row.names = FALSE)
}

if (!interactive()) main()
